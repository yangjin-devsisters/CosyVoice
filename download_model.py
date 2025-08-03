from modelscope import snapshot_download

print("Downloading CosyVoice2-0.5B model...")

# 'iic/CosyVoice2-0.5B' 모델을 'pretrained_models/CosyVoice2-0.5B' 폴더에 다운로드합니다.
snapshot_download(
    'iic/CosyVoice2-0.5B',
    local_dir='pretrained_models/CosyVoice2-0.5B'
)

print("Download complete!")