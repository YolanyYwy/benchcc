import json
from collections import defaultdict

# 读取 JSON 文件
with open(r"/home/yuweiyao/benchcc/data/tau2/domains/retail/tasks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 统计每个 id 出现次数
id_count = defaultdict(int)
for task in data:
    task_id = int(task["id"])
    id_count[task_id] += 1

# 定义要检查的范围
all_ids = range(0, 357)  # 0-356

# 找出缺失的 id
missing_ids = [i for i in all_ids if i not in id_count]

# 找出重复的 id
duplicate_ids = [i for i, count in id_count.items() if count > 1]

print(f"缺失的 ID 总数: {len(missing_ids)}")
print("缺失的 ID 列表:", missing_ids)
print(f"重复的 ID 总数: {len(duplicate_ids)}")
print("重复的 ID 列表:", duplicate_ids)
