import pickle
import os
import subprocess
import yt_dlp

# 配置路径
ANNOTATION_FILE = 'data/annotation-Mar9th-25fps.pkl'
RAW_DIR = 'raw_videos'
CLIPPED_DIR = 'clipped_videos'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLIPPED_DIR, exist_ok=True)

# 加载标注文件
with open(ANNOTATION_FILE, 'rb') as f:
    annotations = pickle.load(f)

def download_and_trim_video(youtube_id, start_time, end_time, output_base):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    raw_output_base = os.path.join(RAW_DIR, output_base)  # 如: raw_videos/abc_30_45
    clipped_output_path = os.path.join(CLIPPED_DIR, f"{output_base}.mp4")

    # 如果裁剪视频已存在，跳过
    if os.path.exists(clipped_output_path):
        print(f"⏭️ 裁剪视频已存在，跳过: {clipped_output_path}")
        return True

    ydl_opts = {
        'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',  # 优先 mp4
        'outtmpl': raw_output_base,  # 注意：不加扩展名，让 yt-dlp 决定
        'quiet': False,
        'no_warnings': False,
        'retries': 3,
        'socket_timeout': 15,
        'cookiefile': 'youtube_cookies.txt',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # 获取 yt-dlp 计算出的真实文件名（含扩展名）
            raw_video_path = ydl.prepare_filename(info)

            # 检查是否已下载
            if os.path.exists(raw_video_path):
                print(f"⏭️ 原始视频已存在，跳过下载: {raw_video_path}")
            else:
                ydl.download([url])
                # 再次确认下载成功
                if not os.path.exists(raw_video_path):
                    print(f"❌ 下载失败：未生成文件 {raw_video_path}")
                    return False

            print(f"✅ 原始视频已保存: {raw_video_path}")

            # 裁剪视频
            duration = end_time - start_time
            cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', raw_video_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-c:a', 'libvo_aacenc',  # 安全，非实验性
                    clipped_output_path
                ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ ffmpeg 裁剪失败: {result.stderr}")
                return False

            print(f"✅ 成功裁剪: {clipped_output_path}")
            return True

    except Exception as e:
        print(f"❌ 处理失败 {url}: {e}")
        return False

# 遍历标注
downloaded_clips = set()

for idx, item in enumerate(annotations):
    meta_info = item[0]
    video_id_with_prefix = meta_info[0]
    youtube_id = video_id_with_prefix.replace('v_', '', 1)
    start_time = meta_info[1]
    end_time = meta_info[2]

    clip_key = (youtube_id, start_time, end_time)
    if clip_key in downloaded_clips:
        continue
    downloaded_clips.add(clip_key)

    output_base = f"{youtube_id}_{int(start_time)}_{int(end_time)}"  # 不加 .mp4
    print(f"🔽 处理 [{idx+1}/{len(annotations)}]: {output_base}")
    download_and_trim_video(youtube_id, start_time, end_time, output_base)

print("🎉 所有视频处理完成！")