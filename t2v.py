import os
import random
import math
import json
import shutil
import pathlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import tensorflow as tf
import keras
from keras import layers
from keras import mixed_precision

from transformers import AutoTokenizer, TFAutoModel

import imageio
import moviepy.editor as mpy
from IPython.display import HTML, display

# Optional: comment out if not using Gradio UI / 可选：若不需要 Gradio UI 可注释掉
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# Expected CUDA/TensorFlow versions for NVIDIA A100 / NVIDIA A100 推荐的 CUDA 与 TensorFlow 版本
print('Recommended: CUDA >= 12.2, cuDNN >= 9, TensorFlow >= 2.14 compiled for CUDA 12.x / 推荐环境配置')
# Verify GPU availability / 检查 GPU 是否可用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = gpus[0].name
    except RuntimeError as err:
        print(f'Failed to set memory growth: {err}')
else:
    DEVICE = 'CPU'

print(f'Active device(s): {[gpu.name for gpu in gpus] if gpus else DEVICE}')
print(f'TensorFlow version: {tf.__version__}')

# Mixed precision policy tailored for A100 float16 Tensor Cores / 针对 A100 Tensor Core 的混合精度策略
mixed_precision.set_global_policy('mixed_float16')
print(f'Mixed precision policy: {mixed_precision.global_policy()}')

# Global seeds for reproducibility / 设定全局随机种子确保结果可复现
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Memory/VRAM assumptions / 默认显存假设
print('Assuming >=40GB VRAM (NVIDIA A100). Adjust batch sizes for smaller GPUs. / 假设显存≥40GB，如显存不足请调小 batch size')
class DatasetConfig(TypedDict):
    video_root: str
    metadata_csv: str
    frames: int
    resolution: Tuple[int, int]
    frame_step: int
    train_split: float


class ModelConfig(TypedDict):
    vae_latent_dim: int
    diffusion_steps: int
    max_time_embeddings: int
    channels: int
    base_channels: int
    channel_multipliers: List[int]
    attention_resolutions: List[int]
    num_heads: int


class TrainingConfig(TypedDict):
    vae_epochs: int
    diffusion_epochs_stage1: int
    diffusion_epochs_stage2: int
    batch_size_stage1: int
    batch_size_stage2: int
    learning_rate_diffusion: float
    learning_rate_vae: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_norm: Optional[float]
    ema_decay: float
    checkpoint_interval: int


class GenerationConfig(TypedDict):
    num_inference_steps: int
    guidance_scale: float
    default_prompts: List[str]


class Config(TypedDict):
    experiment_name: str
    output_dir: str
    log_dir: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig


# Configuration dictionary for experiment tracking / 实验配置字典
CONFIG: Config = {
    'experiment_name': 'text_to_video_diffusion',
    'output_dir': './outputs',
    'log_dir': './logs',
    'dataset': {
        'video_root': '/path/to/video/clips',  # TODO: set dataset path / 需手动填写数据集视频根路径
        'metadata_csv': '/path/to/metadata.csv',  # contains columns [video_path, text, num_frames] / 包含 video_path、text、num_frames 等字段
        'frames': 16,
        'resolution': (64, 64),
        'frame_step': 1,
        'train_split': 0.9,
    },
    'model': {
        'vae_latent_dim': 512,
        'diffusion_steps': 1000,
        'max_time_embeddings': 1024,
        'channels': 3,
        'base_channels': 128,
        'channel_multipliers': [1, 2, 4],
        'attention_resolutions': [4, 8],
        'num_heads': 8,
    },
    'training': {
        'vae_epochs': 30,
        'diffusion_epochs_stage1': 100,
        'diffusion_epochs_stage2': 50,
        'batch_size_stage1': 8,
        'batch_size_stage2': 4,
        'learning_rate_diffusion': 1e-4,
        'learning_rate_vae': 5e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 2000,
        'gradient_clip_norm': 1.0,
        'ema_decay': 0.999,
        'checkpoint_interval': 5,
    },
    'generation': {
        'num_inference_steps': 50,
        'guidance_scale': 7.5,
        'default_prompts': [
            'A serene waterfall cascading into a crystal clear pool at sunset.',
            'A futuristic city skyline illuminated by neon lights during a rainy night.'
        ]
    }
}

pathlib.Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
pathlib.Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
print(json.dumps(CONFIG, indent=2))
# Text tokenizer and encoder utilities / 文本分词与编码工具
class TextEncoder:
    """Wrapper for pretrained Transformer-based text encoder. / 预训练 Transformer 文本编码器封装"""

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModel.from_pretrained(model_name)
        self.embedding_dim = int(self.model.config.hidden_size)

    def __call__(self, texts: List[str]) -> tf.Tensor:
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='tf'
        )
        outputs = self.model(**tokens, training=False)
        # Use pooled output or mean pooling depending on encoder / 根据编码器类型选择池化方式
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
        embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        embeddings_tensor = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        return cast(tf.Tensor, embeddings_tensor)

TEXT_ENCODER = TextEncoder()
print(f'Text encoder loaded: hidden_dim={TEXT_ENCODER.embedding_dim}')
# Dataset utilities / 数据集工具函数
def load_video_clip(path: str, num_frames: int, resolution: Tuple[int, int]) -> np.ndarray:
    """Loads a video file and returns normalized frames in [-1, 1]. Placeholder requires implementation. / 加载视频文件并返回归一化到 [-1, 1] 的帧，需按实际情况完善"""
    frame_step = CONFIG['dataset']['frame_step']
    with imageio.get_reader(path) as reader:
        iter_method = getattr(reader, 'iter_data', None)
        frames: List[np.ndarray] = []
        if callable(iter_method):
            frames_source = iter_method()
            if not isinstance(frames_source, Iterable):
                raise TypeError('iter_data must return an iterable of frames')
            frames_iter = iter(cast(Iterable[np.ndarray], frames_source))
            for idx, frame in enumerate(frames_iter):
                if idx % frame_step == 0:
                    frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
                    resized = tf.image.resize(frame_tensor, resolution)
                    frames.append(np.asarray(resized))
                if len(frames) >= num_frames:
                    break
        else:
            total_frames = num_frames * frame_step
            count_frames_method = getattr(reader, 'count_frames', None)
            if callable(count_frames_method):
                total_candidate = count_frames_method()
                if isinstance(total_candidate, (int, float)):
                    total_frames = int(total_candidate)
            get_data_method = getattr(reader, 'get_data', None)
            if not callable(get_data_method):
                raise AttributeError('Reader object does not provide a callable get_data method.')
            get_data_callable = cast(Callable[[int], Any], get_data_method)
            for frame_index in range(total_frames):
                try:
                    frame = get_data_callable(frame_index)
                except IndexError:
                    break
                if frame_index % frame_step == 0:
                    frame_tensor = tf.convert_to_tensor(np.asarray(frame), dtype=tf.float32)
                    resized = tf.image.resize(frame_tensor, resolution)
                    frames.append(np.asarray(resized))
                if len(frames) >= num_frames:
                    break
    if not frames:
        raise ValueError(f'No frames decoded from {path}. / 无法从视频解码出帧：{path}')
    if len(frames) < num_frames:
        # Loop-pad to reach required length / 若帧数不足则循环补帧
        repeat = num_frames - len(frames)
        frames.extend(frames[:repeat])
    clip = np.stack(frames, axis=0)
    clip = clip.astype('float32')
    clip = (clip / 127.5) - 1.0
    clip = np.transpose(clip, (3, 0, 1, 2))  # [C, T, H, W] / 通道顺序调整为 [通道, 时间, 高, 宽]
    return clip

def load_metadata(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required_cols = {'video_path', 'text'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'Metadata CSV must contain columns: {required_cols}')
    df['video_path'] = df['video_path'].apply(lambda p: os.path.join(CONFIG['dataset']['video_root'], p))
    return df

class VideoTextDataset(keras.utils.Sequence):
    """Keras Sequence for streaming batches of (video, text_embedding) pairs. / 基于 Keras Sequence 的视频-文本批量加载器"""
    
    def __init__(self, dataframe: pd.DataFrame, batch_size: int, num_frames: int, resolution: Tuple[int, int]):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.resolution = resolution
        self.indices = np.arange(len(self.df))
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __getitem__(self, idx: int):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        videos = []
        texts = []
        for i in batch_indices:
            row = self.df.iloc[i]
            clip = load_video_clip(row['video_path'], self.num_frames, self.resolution)
            videos.append(clip)
            texts.append(row['text'])
        video_batch = np.stack(videos, axis=0)  # [B, C, T, H, W] / 批次张量形状
        text_embeddings = TEXT_ENCODER(texts)
        return video_batch, text_embeddings

def create_datasets(metadata_csv: str) -> Tuple[VideoTextDataset, VideoTextDataset]:
    df = load_metadata(metadata_csv)
    split_idx = int(len(df) * CONFIG['dataset']['train_split'])
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    train_dataset = VideoTextDataset(train_df, CONFIG['training']['batch_size_stage1'], CONFIG['dataset']['frames'], CONFIG['dataset']['resolution'])
    val_dataset = VideoTextDataset(val_df, CONFIG['training']['batch_size_stage1'], CONFIG['dataset']['frames'], CONFIG['dataset']['resolution'])
    return train_dataset, val_dataset
# Dataset diagnostics and visualization / 数据集统计与可视化
def dataset_statistics(metadata_csv: str, sample_count: int = 4):
    df = load_metadata(metadata_csv)
    print(f'Total samples: {len(df)} / 数据总量')
    if 'num_frames' in df.columns:
        frame_counts = df['num_frames'].to_numpy()
        print('Average clip length:', frame_counts.mean(), '/ 平均帧数')
        sns.histplot(frame_counts, bins=20)
        plt.title('Frame count distribution / 帧数分布')
        plt.show()
    text_lengths = df['text'].str.split().apply(len).to_numpy()
    sns.histplot(text_lengths, bins=20)
    plt.title('Text length distribution (tokens) / 文本长度分布（词数）')
    plt.show()
    sample_df = df.sample(n=min(sample_count, len(df)), random_state=SEED)
    fig, axes = plt.subplots(1, sample_df.shape[0], figsize=(4 * sample_df.shape[0], 4))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        clip = load_video_clip(row['video_path'], CONFIG['dataset']['frames'], CONFIG['dataset']['resolution'])
        # Display the first frame for a quick glimpse / 可视化第一帧以快速预览
        frame = clip[:, 0]  # [C, H, W] / 张量形状为 [通道, 高, 宽]
        frame = np.transpose(frame, (1, 2, 0))
        frame = ((frame + 1.0) * 127.5).astype('uint8')
        ax.imshow(frame)
        ax.set_title(row['text'][:60] + '...')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return df

# Example usage (uncomment after setting metadata path) / 示例：配置路径后取消注释
# dataset_statistics(CONFIG['dataset']['metadata_csv'])
# Latent Video VAE (optional but recommended) / 视频潜空间 VAE（可选但推荐）
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_video_vae(input_shape: Tuple[int, int, int, int], latent_dim: int):
    channels, frames, height, width = input_shape
    encoder_inputs = keras.Input(shape=(channels, frames, height, width), name='encoder_input')
    x = layers.Permute((2, 3, 4, 1))(encoder_inputs)  # [T, H, W, C] / 维度换位方便 3D 卷积
    x = layers.Conv3D(64, 3, strides=(1, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3D(128, 3, strides=(2, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3D(256, 3, strides=(2, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='video_encoder')

    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense((frames // 4) * (height // 8) * (width // 8) * 256)(latent_inputs)
    x = layers.Reshape((frames // 4, height // 8, width // 8, 256))(x)
    x = layers.Conv3DTranspose(256, 3, strides=(2, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3DTranspose(128, 3, strides=(2, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv3DTranspose(64, 3, strides=(1, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    decoder_outputs = layers.Conv3DTranspose(channels, 3, activation='tanh', padding='same', name='decoder_output')(x)
    decoder_outputs = layers.Permute((4, 1, 2, 3))(decoder_outputs)

    decoder = keras.Model(latent_inputs, decoder_outputs, name='video_decoder')

    class VideoVAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
            self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            return reconstruction

        def train_step(self, data):
            videos, _ = data
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(videos, training=True)
                reconstruction = self.decoder(z, training=True)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(videos - reconstruction), axis=[1, 2, 3, 4]))
                kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            if grads is None:
                raise RuntimeError('Gradient computation returned None for VAE model.')
            if self.optimizer is None:
                raise RuntimeError('Optimizer must be set before calling train_step.')
            grad_var_pairs: List[Tuple[tf.Tensor, tf.Variable]] = []
            for grad, var in zip(grads, self.trainable_weights):
                if grad is None:
                    continue
                grad_tensor = cast(tf.Tensor, grad)
                grad_var_pairs.append((grad_tensor, var))
            self.optimizer.apply_gradients(grad_var_pairs)
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                'loss': self.total_loss_tracker.result(),
                'reconstruction_loss': self.reconstruction_loss_tracker.result(),
                'kl_loss': self.kl_loss_tracker.result(),
            }

    vae = VideoVAE(encoder, decoder, name='video_vae')
    return vae

VAE_MODEL = build_video_vae((CONFIG['model']['channels'], CONFIG['dataset']['frames'], *CONFIG['dataset']['resolution']), CONFIG['model']['vae_latent_dim'])
vae_optimizer: Any = keras.optimizers.Adam(learning_rate=float(CONFIG['training']['learning_rate_vae']))
VAE_MODEL.compile(optimizer=vae_optimizer)
VAE_MODEL.summary()
# Temporal attention and UNet building blocks / 时间注意力与 UNet 组件
def sinusoidal_time_embedding(timesteps: tf.Tensor, dim: int) -> tf.Tensor:
    timesteps = tf.convert_to_tensor(timesteps)
    half_dim = dim // 2
    batch_shape = tf.shape(timesteps)
    batch_size = tf.gather(batch_shape, 0)
    if half_dim == 0:
        return tf.zeros((batch_size, dim), dtype=tf.float32)
    denominator = tf.cast(tf.maximum(half_dim - 1, 1), tf.float32)
    log_term = tf.math.log(tf.constant(10000.0, dtype=tf.float32))
    exponent = -log_term / denominator
    frequencies = tf.exp(tf.range(half_dim, dtype=tf.float32) * exponent)
    angles = tf.cast(tf.reshape(timesteps, (-1, 1)), tf.float32) * tf.reshape(frequencies, (1, -1))
    emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    emb = tf.cast(emb, dtype=tf.float32)
    return tf.ensure_shape(emb, [None, dim])

class ResidualBlock(layers.Layer):
    def __init__(self, channels: int, time_emb_dim: int, use_attention: bool = False, num_heads: int = 4, name: str = 'res_block'):
        super().__init__(name=name)
        self.channels = channels
        self.time_mlp = keras.Sequential([layers.Dense(channels, activation='gelu'), layers.Dense(channels)])
        self.conv1 = layers.Conv3D(channels, 3, padding='same')
        self.conv2 = layers.Conv3D(channels, 3, padding='same')
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)
        self.act = layers.Activation('gelu')
        self.use_attention = use_attention
        if use_attention:
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(channels // num_heads, 1))
            self.attn_norm = layers.LayerNormalization(axis=-1)
        self.proj: Optional[layers.Layer] = None

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.proj = layers.Conv3D(self.channels, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs, time_emb: tf.Tensor):
        h = self.norm1(inputs)
        h = self.act(h)
        h = self.conv1(h)
        time_emb_proj = self.time_mlp(time_emb)
        time_emb_proj = tf.reshape(time_emb_proj, (-1, 1, 1, 1, self.channels))
        h = h + time_emb_proj
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        if self.use_attention:
            shape_tensor = tf.shape(h)
            shape_components = cast(List[tf.Tensor], tf.unstack(shape_tensor, num=5))
            batch, time_steps, height, width, channels = shape_components
            spatial = tf.math.multiply(height, width)
            flattened = tf.reshape(h, tf.stack([batch, time_steps, spatial, channels]))
            attn_out = self.attn(flattened, flattened)
            attn_out = self.attn_norm(attn_out)
            attn_out = tf.reshape(attn_out, tf.stack([batch, time_steps, height, width, channels]))
            h = h + attn_out
        residual = inputs if self.proj is None else self.proj(inputs)
        return residual + h

class Downsample(layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = layers.Conv3D(channels, 3, strides=(1, 2, 2), padding='same')
    def call(self, x):
        return self.conv(x)

class Upsample(layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = layers.Conv3DTranspose(channels, 3, strides=(1, 2, 2), padding='same')
    def call(self, x):
        return self.conv(x)
# Video UNet conditioned on text embeddings / 文本条件视频 UNet
class VideoUNet(keras.Model):
    def __init__(self, base_channels: int, channel_mults: List[int], attention_resolutions: List[int], num_heads: int, time_emb_dim: int, text_emb_dim: int):
        super().__init__(name='video_unet')
        self.time_mlp = keras.Sequential([
            layers.Dense(time_emb_dim, activation='gelu'),
            layers.Dense(time_emb_dim)
        ])
        self.text_proj = layers.Dense(time_emb_dim)
        self.input_conv = layers.Conv3D(base_channels, 3, padding='same')
        self.downs: List[List[layers.Layer]] = []
        self.ups: List[List[layers.Layer]] = []
        in_channels = base_channels
        resolution = CONFIG['dataset']['resolution'][0]
        for mult in channel_mults:
            out_channels = base_channels * mult
            use_attention = resolution in attention_resolutions
            self.downs.append([
                ResidualBlock(out_channels, time_emb_dim, use_attention=use_attention, num_heads=num_heads),
                ResidualBlock(out_channels, time_emb_dim, use_attention=use_attention, num_heads=num_heads),
                Downsample(out_channels)
            ])
            in_channels = out_channels
            resolution //= 2
        self.mid_block1 = ResidualBlock(in_channels, time_emb_dim, use_attention=True, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(in_channels, time_emb_dim, use_attention=False, num_heads=num_heads)
        for mult in reversed(channel_mults):
            out_channels = base_channels * mult
            use_attention = resolution in attention_resolutions
            self.ups.append([
                Upsample(out_channels),
                ResidualBlock(out_channels, time_emb_dim, use_attention=use_attention, num_heads=num_heads),
                ResidualBlock(out_channels, time_emb_dim, use_attention=use_attention, num_heads=num_heads)
            ])
            resolution *= 2
        self.output_conv = layers.Conv3D(CONFIG['model']['channels'], 3, padding='same')

    def call(self, x, timesteps, text_embeddings, training=False):
        t_emb = sinusoidal_time_embedding(timesteps, CONFIG['model']['max_time_embeddings'])
        t_emb = self.time_mlp(t_emb)
        txt_emb = self.text_proj(text_embeddings)
        combined_emb = tf.nn.gelu(t_emb + txt_emb)
        h = self.input_conv(x)
        skips = []
        for res_blocks in self.downs:
            for block in res_blocks[:-1]:
                h = block(h, combined_emb)
                skips.append(h)
            h = res_blocks[-1](h)
        h = self.mid_block1(h, combined_emb)
        h = self.mid_block2(h, combined_emb)
        for res_blocks in self.ups:
            upsample, block1, block2 = res_blocks
            h = upsample(h)
            skip = skips.pop()
            h = tf.concat([h, skip], axis=-1)
            h = block1(h, combined_emb)
            skip = skips.pop()
            h = tf.concat([h, skip], axis=-1)
            h = block2(h, combined_emb)
        output = self.output_conv(h)
        return output
# Gaussian diffusion schedule and model wrapper / 高斯扩散调度与模型封装
class GaussianDiffusion:
    def __init__(self, timesteps: int, beta_schedule: str = 'cosine'):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = np.linspace(1e-4, 0.02, timesteps, dtype=np.float32)
        elif beta_schedule == 'cosine':
            steps = np.arange(timesteps + 1, dtype=np.float64) / timesteps
            alphas_cumprod = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 1e-5, 0.999)
        else:
            raise ValueError('Unsupported beta schedule')
        self.betas = tf.constant(betas, dtype=tf.float32)
        ones = tf.ones_like(self.betas)
        self.alphas = ones - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = tf.concat([tf.constant([1.0], dtype=tf.float32), self.alphas_cumprod[:-1]], axis=0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(tf.ones_like(self.alphas_cumprod) - self.alphas_cumprod)
        self.posterior_variance = self.betas * (tf.ones_like(self.alphas_cumprod_prev) - self.alphas_cumprod_prev) / (tf.ones_like(self.alphas_cumprod) - self.alphas_cumprod)

    def q_sample(self, x_start: tf.Tensor, t: tf.Tensor, noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))
        sqrt_alphas_cumprod_t = tf.reshape(tf.gather(self.sqrt_alphas_cumprod, t), (-1, 1, 1, 1, 1))
        sqrt_one_minus = tf.reshape(tf.gather(self.sqrt_one_minus_alphas_cumprod, t), (-1, 1, 1, 1, 1))
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus * noise

    def p_sample(self, denoise_fn, x, t, text_embeddings, guidance_scale: float = 1.0):
        beta_t = tf.reshape(tf.gather(self.betas, t), (-1, 1, 1, 1, 1))
        alpha_t = tf.reshape(tf.gather(self.alphas, t), (-1, 1, 1, 1, 1))
        sqrt_one_minus = tf.reshape(tf.gather(self.sqrt_one_minus_alphas_cumprod, t), (-1, 1, 1, 1, 1))
        sqrt_recip_alpha = tf.math.rsqrt(alpha_t)
        z = tf.random.normal(tf.shape(x))
        eps = denoise_fn(x, t, text_embeddings)
        if guidance_scale != 1.0:
            eps = eps * guidance_scale
        x0_pred = (x - sqrt_one_minus * eps) * sqrt_recip_alpha
        dir_xt = tf.sqrt(tf.ones_like(beta_t) - beta_t) * x
        noise_term = tf.sqrt(beta_t) * z
        return dir_xt + noise_term, x0_pred

DIFFUSION = GaussianDiffusion(CONFIG['model']['diffusion_steps'])
# Diffusion model wrapper integrating UNet and training logic / 融合 UNet 与训练逻辑的扩散模型
class VideoDiffusionModel(keras.Model):
    def __init__(self, unet: VideoUNet, diffusion: GaussianDiffusion):
        super().__init__(name='video_diffusion')
        self.unet = unet
        self.diffusion = diffusion
        self.loss_tracker = keras.metrics.Mean(name='diffusion_loss')

    def call(self, inputs, training=False):
        videos, text_embeddings = inputs
        video_shape = tf.shape(videos, out_type=tf.int32)
        batch_size = tf.gather(video_shape, 0)
        timesteps = tf.random.uniform(
            shape=tf.reshape(batch_size, (1,)),
            minval=0,
            maxval=self.diffusion.timesteps,
            dtype=tf.int32,
        )
        timesteps = tf.reshape(timesteps, (-1,))
        noise = tf.random.normal(tf.shape(videos))
        noisy_videos = self.diffusion.q_sample(videos, timesteps, noise)
        pred_noise = self.unet(noisy_videos, timesteps, text_embeddings, training=training)
        loss = tf.reduce_mean(tf.square(noise - pred_noise))
        if training:
            self.add_loss(loss)
        return loss

    @tf.function
    def train_step(self, data):
        videos, text_embeddings = data
        clip_norm = CONFIG['training']['gradient_clip_norm']
        with tf.GradientTape() as tape:
            loss = self(videos, text_embeddings, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.optimizer is None:
            raise RuntimeError('Optimizer must be set before training.')
        if gradients is None:
            raise RuntimeError('Gradient computation returned None for diffusion model.')
        grad_var_pairs: List[Tuple[tf.Tensor, tf.Variable]] = []
        for grad, var in zip(gradients, self.trainable_variables):
            if grad is None:
                continue
            grad_tensor = cast(tf.Tensor, grad)
            if clip_norm is not None:
                grad_tensor = cast(tf.Tensor, tf.clip_by_norm(grad_tensor, clip_norm))
            grad_var_pairs.append((grad_tensor, var))
        self.optimizer.apply_gradients(grad_var_pairs)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def sample(self, text_embeddings: tf.Tensor, num_frames: int, resolution: Tuple[int, int], num_steps: int, guidance_scale: float = 1.0):
        text_shape = tf.shape(text_embeddings, out_type=tf.int32)
        batch_size = tf.gather(text_shape, 0)
        batch_shape = tf.reshape(batch_size, (1,))
        shape_tail = tf.constant([CONFIG['model']['channels'], num_frames, resolution[0], resolution[1]], dtype=tf.int32)
        sample_shape = tf.concat([batch_shape, shape_tail], axis=0)
        x = tf.random.normal(sample_shape)
        latest_x0 = x
        dims = tf.reshape(batch_size, (1,))
        for step in reversed(range(num_steps)):
            timestep_tensor = tf.fill(dims, tf.cast(step, tf.int32))
            x, latest_x0 = self.diffusion.p_sample(self.unet, x, timestep_tensor, text_embeddings, guidance_scale)
        return x, latest_x0

UNET = VideoUNet(
    base_channels=CONFIG['model']['base_channels'],
    channel_mults=CONFIG['model']['channel_multipliers'],
    attention_resolutions=CONFIG['model']['attention_resolutions'],
    num_heads=CONFIG['model']['num_heads'],
    time_emb_dim=CONFIG['model']['max_time_embeddings'],
    text_emb_dim=TEXT_ENCODER.embedding_dim
)
diffusion_optimizer: Any = keras.optimizers.AdamW(
    learning_rate=float(CONFIG['training']['learning_rate_diffusion']),
    weight_decay=float(CONFIG['training']['weight_decay'])
 )
DIFFUSION_MODEL = VideoDiffusionModel(UNET, DIFFUSION)
DIFFUSION_MODEL.compile(optimizer=diffusion_optimizer)
DIFFUSION_MODEL.summary()
class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, dtype=tf.float32)
        self.warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
        self.total_steps = tf.convert_to_tensor(total_steps, dtype=tf.float32)

    def __call__(self, step: tf.Tensor | float | int) -> tf.Tensor:
        step_tensor = tf.convert_to_tensor(step, dtype=tf.float32)
        zero_scalar = tf.constant(0.0, dtype=tf.float32)
        warmup_positive = tf.math.greater(self.warmup_steps, zero_scalar)
        warmup_divisor = tf.where(warmup_positive, self.warmup_steps, tf.ones_like(self.warmup_steps))
        warmup_ratio = tf.clip_by_value(tf.math.divide(step_tensor, warmup_divisor), 0.0, 1.0)
        warmup_lr = tf.math.multiply(self.base_lr, warmup_ratio)
        pi_const = tf.constant(math.pi, dtype=tf.float32)
        total_positive = tf.math.greater(self.total_steps, zero_scalar)
        total_steps_safe = tf.where(total_positive, self.total_steps, tf.ones_like(self.total_steps))
        cosine_argument = tf.math.divide(tf.math.multiply(pi_const, step_tensor), total_steps_safe)
        half = tf.constant(0.5, dtype=tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)
        cosine_decay = tf.math.multiply(half, tf.math.add(one, tf.cos(cosine_argument)))
        cosine_lr = tf.math.multiply(self.base_lr, cosine_decay)
        use_warmup = tf.math.less(step_tensor, self.warmup_steps)
        return tf.where(use_warmup, warmup_lr, cosine_lr)

class ExponentialMovingAverage:
    def __init__(self, model: keras.Model, decay: float = 0.999):
        self.decay = decay
        self.shadow_vars = [tf.Variable(w, trainable=False) for w in model.weights]
        self.model = model
        self.backup: List[tf.Tensor] = []

    def update(self):
        for shadow, weight in zip(self.shadow_vars, self.model.weights):
            decay_tensor = tf.cast(tf.convert_to_tensor(self.decay, dtype=shadow.dtype), shadow.dtype)
            one_minus = tf.cast(tf.convert_to_tensor(1.0 - self.decay, dtype=shadow.dtype), shadow.dtype)
            shadow_tensor = tf.cast(tf.convert_to_tensor(shadow), shadow.dtype)
            weight_tensor = tf.cast(tf.convert_to_tensor(weight), shadow.dtype)
            updated_shadow = tf.add(tf.multiply(shadow_tensor, decay_tensor), tf.multiply(weight_tensor, one_minus))
            shadow.assign(updated_shadow)

    def apply_ema_weights(self):
        self.backup = [cast(tf.Tensor, tf.cast(tf.convert_to_tensor(weight), weight.dtype)) for weight in self.model.weights]
        for weight, shadow in zip(self.model.weights, self.shadow_vars):
            weight.assign(tf.cast(shadow, weight.dtype))

    def restore_weights(self):
        for weight, backup_weight in zip(self.model.weights, self.backup):
            weight.assign(tf.cast(backup_weight, weight.dtype))

def setup_tensorboard(log_dir: str, name: str):
    log_path = os.path.join(log_dir, name)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(log_path)
    return writer

def save_checkpoint(model: keras.Model, optimizer: keras.optimizers.Optimizer, epoch: int, prefix: str):
    checkpoint_path = os.path.join(CONFIG['output_dir'], f'{prefix}_epoch_{epoch}.weights.h5')
    model.save_weights(checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path} / 已保存检查点')
# Stage-wise training functions / 分阶段训练流程
def train_vae(train_dataset: VideoTextDataset, val_dataset: VideoTextDataset):
    writer = setup_tensorboard(CONFIG['log_dir'], 'vae')
    for epoch in range(CONFIG['training']['vae_epochs']):
        print(f"VAE Epoch {epoch + 1}/{CONFIG['training']['vae_epochs']} / VAE 训练轮次")
        for batch_index in tqdm(range(len(train_dataset)), desc='VAE Train / VAE 训练'):
            videos, _ = train_dataset[batch_index]
            VAE_MODEL.train_on_batch(videos, videos)
        val_losses = []
        for batch_index in range(len(val_dataset)):
            videos, _ = val_dataset[batch_index]
            val_result = VAE_MODEL.evaluate(videos, videos, verbose='auto')
            val_loss = val_result['loss'] if isinstance(val_result, dict) else val_result
            val_losses.append(float(val_loss))
        with writer.as_default():
            tf.summary.scalar('val_loss', np.mean(val_losses), step=epoch)
    return VAE_MODEL

def train_diffusion(train_dataset: VideoTextDataset, val_dataset: VideoTextDataset, stage: str = 'stage1'):
    num_epochs = CONFIG['training']['diffusion_epochs_stage1'] if stage == 'stage1' else CONFIG['training']['diffusion_epochs_stage2']
    ema = ExponentialMovingAverage(DIFFUSION_MODEL, CONFIG['training']['ema_decay'])
    writer = setup_tensorboard(CONFIG['log_dir'], f'diffusion_{stage}')
    global_step = 0
    total_steps = num_epochs * max(len(train_dataset), 1)
    lr_schedule = WarmupCosineSchedule(
        base_lr=CONFIG['training']['learning_rate_diffusion'],
        warmup_steps=CONFIG['training']['warmup_steps'],
        total_steps=total_steps
    )
    for epoch in range(num_epochs):
        with writer.as_default():
            tf.summary.scalar('learning_rate', lr_schedule(global_step), step=global_step)
        print(f'Diffusion {stage} Epoch {epoch + 1}/{num_epochs} / 扩散阶段 {stage} 轮次')
        optimizer = DIFFUSION_MODEL.optimizer
        if optimizer is None:
            raise RuntimeError('Diffusion model must be compiled with an optimizer.')
        for batch_index in tqdm(range(len(train_dataset)), desc='Diffusion Train / 扩散模型训练'):
            lr_value = cast(tf.Tensor, lr_schedule(global_step))
            if isinstance(optimizer.learning_rate, tf.Variable):
                optimizer.learning_rate.assign(lr_value)
            videos, text_embeddings = train_dataset[batch_index]
            train_result = DIFFUSION_MODEL.train_on_batch(videos, text_embeddings)
            train_loss = train_result['loss'] if isinstance(train_result, dict) else train_result
            ema.update()
            if global_step % 50 == 0:
                with writer.as_default():
                    tf.summary.scalar('train_loss', tf.convert_to_tensor(train_loss), step=global_step)
            global_step += 1
        val_losses = []
        for batch_index in range(len(val_dataset)):
            videos, text_embeddings = val_dataset[batch_index]
            val_result = DIFFUSION_MODEL.evaluate(videos, text_embeddings, verbose='auto')
            val_loss = val_result['loss'] if isinstance(val_result, dict) else val_result
            val_losses.append(float(val_loss))
        with writer.as_default():
            tf.summary.scalar('val_loss', np.mean(val_losses), step=global_step)
        if (epoch + 1) % CONFIG['training']['checkpoint_interval'] == 0:
            save_checkpoint(DIFFUSION_MODEL, optimizer, epoch + 1, stage)
    return DIFFUSION_MODEL, ema
# Generation utilities / 视频生成工具
def latent_to_video(latents: tf.Tensor) -> np.ndarray:
    """Convert latent or pixel-space tensors to uint8 video arrays. / 将潜变量或像素空间张量转为 uint8 视频帧"""
    tensor = tf.convert_to_tensor(latents)
    if len(tensor.shape) == 2 and VAE_MODEL is not None:
        reconstruction = cast(tf.Tensor, VAE_MODEL.decoder(tensor, training=False))
    else:
        reconstruction = tensor
    reconstruction_tensor = tf.convert_to_tensor(reconstruction)
    video = tf.transpose(reconstruction_tensor, perm=[0, 2, 3, 4, 1])
    video = tf.clip_by_value((video + 1.0) * 127.5, 0.0, 255.0)
    video_uint8 = tf.cast(video, tf.uint8)
    return np.asarray(tf.convert_to_tensor(video_uint8))

def save_video(frames: np.ndarray, path: str, fps: int = 8, fmt: str = 'mp4'):
    frames_list = list(frames)
    if fmt == 'gif':
        imageio.mimsave(path, frames_list, fps=fps)
    else:
        clip = mpy.ImageSequenceClip(frames_list, fps=fps)
        clip.write_videofile(path, codec='libx264', audio=False, verbose=False, logger=None)

def display_video(frames: np.ndarray, fps: int = 8):
    clip = mpy.ImageSequenceClip(list(frames), fps=fps)
    video_html = clip.to_html5_video()  # type: ignore[attr-defined]
    display(HTML(video_html))

def generate_video(prompt: str, num_frames: int = 16, resolution: Tuple[int, int] = (64, 64), batch_size: int = 1, num_steps: Optional[int] = None, guidance_scale: Optional[float] = None, save_path: Optional[str] = None) -> np.ndarray:
    steps = num_steps if num_steps is not None else CONFIG['generation']['num_inference_steps']
    guidance = guidance_scale if guidance_scale is not None else CONFIG['generation']['guidance_scale']
    if len(resolution) != 2:
        raise ValueError('Resolution must be a (height, width) tuple of length 2.')
    height, width = int(resolution[0]), int(resolution[1])
    text_embeddings = TEXT_ENCODER([prompt] * batch_size)
    _, predictions = DIFFUSION_MODEL.sample(text_embeddings, num_frames, (height, width), int(steps), float(guidance))
    videos = latent_to_video(predictions)
    for idx, video in enumerate(videos):
        if save_path and batch_size == 1:
            target_path = save_path
        else:
            base_dir = os.path.dirname(save_path) if save_path else CONFIG['output_dir']
            pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
            target_path = os.path.join(base_dir, f'generated_{idx}.mp4')
        save_video(video, target_path, fmt='mp4')
        print(f'Saved video to {target_path} / 视频已保存')
        display_video(video)
    return videos

# Example usage (after training) / 训练完成后的示例用法
# generate_video('A tranquil lake surrounded by snowy mountains during sunrise.')
# Evaluation: plotting losses, qualitative samples, FVD placeholder / 评估：损失曲线、案例展示与 FVD 占位
def plot_loss_curves(log_dir: str, run_name: str):
    event_path = os.path.join(log_dir, run_name)
    print(f'Use TensorBoard to inspect logs: tensorboard --logdir {event_path} / 使用 TensorBoard 查看日志')

def compare_prompts(prompts: List[str]):
    fig, axes = plt.subplots(len(prompts), 1, figsize=(6, 4 * len(prompts)))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for ax, prompt in zip(axes, prompts):
        videos = generate_video(prompt, save_path=os.path.join(CONFIG['output_dir'], f"{prompt[:20].replace(' ', '_')}.mp4"))
        ax.imshow(videos[0][0])
        ax.set_title(prompt)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def compute_fvd_placeholder(real_videos: np.ndarray, generated_videos: np.ndarray) -> float:
    # TODO: Integrate official FVD computation (requires I3D features). / 待办：集成官方 FVD 计算（需 I3D 特征）
    # Placeholder returns NaN and reminds to implement. / 目前返回 NaN 作为提醒
    return float('nan')

def ablation_experiment(prompts: List[str], frame_counts: List[int], resolutions: List[Tuple[int, int]], use_attention: List[bool]):
    results = []
    for prompt in prompts:
        for frames in frame_counts:
            for res in resolutions:
                for attention in use_attention:
                    if not attention:
                        # Temporarily disable attention by adjusting UNET parameters / 临时关闭注意力以做对比实验
                        original_attention = CONFIG['model']['attention_resolutions']
                        CONFIG['model']['attention_resolutions'] = []
                    videos = generate_video(prompt, num_frames=frames, resolution=res)
                    results.append({'prompt': prompt, 'frames': frames, 'resolution': res, 'attention': attention})
                    if not attention:
                        CONFIG['model']['attention_resolutions'] = original_attention
    return pd.DataFrame(results)
# Optional Gradio interface for interactive demos / 可选的 Gradio 交互式演示界面
def setup_gradio_interface():
    if not GRADIO_AVAILABLE:
        raise ImportError('Gradio not installed. Run `pip install gradio`. / 未安装 Gradio，请先运行 pip install gradio')

    def infer(prompt, frames, resolution):
        res_values = tuple(map(int, resolution.split('x')))
        if len(res_values) != 2:
            raise ValueError('Resolution dropdown must provide values like 64x64. / Gradio 分辨率格式需为 如 64x64')
        res_tuple: Tuple[int, int] = (res_values[0], res_values[1])
        videos = generate_video(prompt, num_frames=int(frames), resolution=res_tuple)
        temp_path = os.path.join(CONFIG['output_dir'], 'gradio_preview.mp4')
        save_video(videos[0], temp_path)
        return temp_path

    with gr.Blocks() as demo:
        gr.Markdown('# Text-to-Video Diffusion Demo / 文本生成视频实时演示')
        prompt = gr.Textbox(label='Prompt / 文本提示', value='A paper boat floating down a rainy street at night.')
        frames = gr.Slider(8, 32, value=16, step=1, label='Frames / 帧数')
        resolution = gr.Dropdown(['64x64', '96x96', '128x128'], value='64x64', label='Resolution / 分辨率')
        btn = gr.Button('Generate / 生成')
        output = gr.Video(label='Generated Video / 生成视频')
        btn.click(fn=infer, inputs=[prompt, frames, resolution], outputs=output)
    return demo

# To launch: demo = setup_gradio_interface(); demo.launch(share=False) / 启动方法如上