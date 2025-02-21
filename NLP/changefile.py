import argparse
import re

def modify_file(file_path, new_cola_value, new_list):
    """
    修改源文件中 glue_tasks_num_labels 的 "cola" 值，并替换带 #mark 的 return 列表。

    :param file_path: str, 需要修改的文件路径
    :param new_cola_value: int, 替换后的 "cola" 的值
    :param new_list: list[str], 替换 return 语句的新列表
    """
    try:
        # 打开并读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # 修改 glue_tasks_num_labels 中的 "cola" 值
        pattern_cola = r'"cola":\s*\d+'
        replacement_cola = f'"cola": {new_cola_value}'
        updated_content = re.sub(pattern_cola, replacement_cola, file_content)

        # 替换带 #mark 注释的 return 语句
        pattern_return = r'return\s*\[.*?\]\s*#mark'
        replacement_return = f'return {new_list} #mark'
        updated_content = re.sub(pattern_return, replacement_return, updated_content)

        # 覆盖保存修改后的内容到原文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        print(f"文件已更新: {file_path}")
        print(f"- 'cola' 的值已修改为 {new_cola_value}")
        print(f"- return 列表已修改为 {new_list}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

def getreplacement(args):
    if args.dataset_name == 'PHEE_EE':
        new_cola_value = 2
        new_list = "['0', '1']"
    elif args.dataset_name == 'DDI':
        new_cola_value = 5
        new_list = '["0", "1", "2", "3", "4"]'
    elif args.dataset_name == 'medical_meadow_health_advice':
        new_cola_value = 3
        new_list = '["0", "1", "2"]'
    elif args.dataset_name == 'MIMIC':
        new_cola_value = 4
        new_list = '["0", "1", "2", "3"]'
    return new_cola_value,new_list

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="修改 glue_tasks_num_labels 和带 #mark 的 return 语句")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    args = parser.parse_args()
    
    new_cola_value,new_list = getreplacement(args)
    file_path = "./src/transformers/data/processors/glue.py"

    # 调用函数执行修改
    modify_file(
        file_path=file_path,
        new_cola_value=new_cola_value,
        new_list=new_list
    )
