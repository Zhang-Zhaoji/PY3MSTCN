from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

root_dir = Path('./model')

# 递归查找所有 event 文件
event_files = list(root_dir.rglob("events.out.tfevents.*"))
print(f"Found {len(event_files)} TensorBoard event files.")

results = []

for event_file in sorted(event_files):
    print(f"\n🔍 Processing: {event_file.relative_to(root_dir)}")
    try:
        loader = EventFileLoader(str(event_file))
        graph_def = None
        for event in loader.Load():
            if event.HasField("graph_def") and event.graph_def:
                graph_def = event.graph_def
                break  # 通常只有一个 graph

        if graph_def is None:
            print("  ⚠️  No graph_def found in event file.")
            results.append((str(event_file), []))
            continue

        # 将 graph_def 转为可搜索的字符串
        graph_str = graph_def.decode('utf-8', errors='ignore').lower()
        stages = []
        if 'squeeze2' in graph_str:
            stages.append('Squeeze2Stage')
        if 'squeeze1' in graph_str:
            stages.append('Squeeze1Stage')
        if 'squeeze0' in graph_str:
            stages.append('Squeeze0Stage')

        if stages:
            print(f"  ✅ Detected: {', '.join(stages)}")
        else:
            # 可选：打印一小段用于调试
            print("  ℹ️  No keywords found. Sample snippet:")
            print("      " + graph_str[:300].replace('\n', ' ') + " ...")

        results.append((str(event_file), stages))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append((str(event_file), None))

# 汇总
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for path, stages in results:
    label = (
        "(error)" if stages is None else
        ", ".join(stages) if stages else
        "(none)"
    )
    print(f"{path} → {label}")