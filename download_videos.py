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

# 已下载的 YouTube ID 集合
downloaded_videos = set()

def download_full_video(youtube_id):
    """下载完整原始视频，只下载一次"""
    raw_video_path = os.path.join(RAW_DIR, f"{youtube_id}.mp4")

    # 如果已下载，跳过
    if os.path.exists(raw_video_path):
        print(f"⏭️ 完整视频已存在，跳过下载: {raw_video_path}")
        return raw_video_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': raw_video_path,  # 保存为 youtube_id.mp4
        'cookiefile': 'youtube_cookies.txt',  # 如果你有 cookies
        'quiet': False,
        'no_warnings': False,
        'retries': 3,
        'socket_timeout': 15,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)

            # 下载
            ydl.download([url])

            # yt-dlp 可能保存为 .webm 等，但我们期望 .mp4
            # 如果文件存在但不是 .mp4，尝试重命名或转码（这里简单处理）
            if not os.path.exists(raw_video_path) and os.path.exists(filename):
                os.rename(filename, raw_video_path)
                print(f"🔄 重命名为: {raw_video_path}")
            elif os.path.exists(raw_video_path):
                pass
            else:
                print(f"❌ 下载失败：未生成 {raw_video_path} 或 {filename}")
                return None

        print(f"✅ 完整视频已下载: {raw_video_path}")
        return raw_video_path

    except Exception as e:
        print(f"❌ 下载完整视频失败 {url}: {e}")
        return None

def trim_clip(raw_video_path, start_time, end_time, output_path):
    """从完整视频中裁剪片段"""
    if os.path.exists(output_path):
        print(f"⏭️ 裁剪视频已存在，跳过: {output_path}")
        return True

    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', raw_video_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'libvo_aacenc',  # 使用非实验性编码器
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ ffmpeg 裁剪失败: {result.stderr}")
            return False
        print(f"✅ 成功裁剪: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 裁剪出错: {e}")
        return False

# 主循环
for idx, item in enumerate(annotations):
    meta_info = item[0]
    video_id_with_prefix = meta_info[0]
    youtube_id = video_id_with_prefix.replace('v_', '', 1)
    start_time = meta_info[1]
    end_time = meta_info[2]

    # 裁剪输出文件
    clip_filename = f"{youtube_id}_{int(start_time)}_{int(end_time)}.mp4"
    clipped_output_path = os.path.join(CLIPPED_DIR, clip_filename)

    # 如果裁剪视频已存在，跳过
    if os.path.exists(clipped_output_path):
        print(f"⏭️ 裁剪视频已存在，跳过: {clipped_output_path}")
        continue

    # 下载完整视频（仅一次）
    if youtube_id not in downloaded_videos:
        video_path = download_full_video(youtube_id)
        if not video_path:
            print(f"❌ 无法下载视频 {youtube_id}，跳过所有相关片段")
            downloaded_videos.add(youtube_id)  # 避免重复尝试
            continue
        downloaded_videos.add(youtube_id)
    else:
        video_path = os.path.join(RAW_DIR, f"{youtube_id}.mp4")
        if not os.path.exists(video_path):
            print(f"❌ 视频文件丢失: {video_path}")
            continue

    # 裁剪片段
    print(f"🔽 正在裁剪 [{idx+1}/{len(annotations)}]: {clip_filename}")
    trim_clip(video_path, start_time, end_time, clipped_output_path)

print("🎉 所有视频处理完成！")