import json

# 读取第一个文件
with open(r"D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

# 读取第二个文件
with open(r"D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks_augmented.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

# 拼接数组
merged = data1 + data2

# 写入合并后的文件
with open(r"D:\Desktop\work\tau2-bench\data\tau2\domains\retail\merged.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)

print("Merged!")
