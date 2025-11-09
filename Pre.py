"""
从Excel文件读取订单信息并转换为FJSP求解所需的数据格式
"""

import pandas as pd
import os
from typing import List, Dict, Tuple

class OrderProcessor:
    def __init__(self, excel_folder='dingdan'):
        """
        初始化订单处理器
        
        Args:
            excel_folder: 存放Excel订单文件的文件夹路径或单个文件路径
        """
        self.excel_folder = excel_folder
        self.orders = []  # 存储所有订单
        self.processing_time = []  # 最终的加工时间矩阵(含调机)
        self.processing_time_no_setup = []  # 不含调机的加工时间矩阵
        self.jobs_dict = {}  # 每个工件的工序数
        self.job_metadata = {}  # 工件元数据
        self.total_operations = 0  # 总工序数
        self.max_machines = 50  # 最大机器数(设置为50以支持更多机器)
        self.actual_max_machine = 0  # 实际使用的最大机器编号
        
    def read_excel_file(self, filepath: str) -> Dict:
        """读取单个Excel文件"""
        print(f"\n读取文件: {os.path.basename(filepath)}")
        
        try:
            # 读取Excel,不设置表头
            df = pd.read_excel(filepath, header=None)
            
            # 读取基本信息(前3行)
            order_num = str(df.iloc[0, 0])  # 订单号
            part_num = str(df.iloc[1, 0])   # 工件号
            quantity = int(df.iloc[2, 0])   # 工件数量
            
            print(f"  订单号: {order_num}, 工件号: {part_num}, 数量: {quantity}")
            
            # 读取工序信息(第4行起,即索引3开始)
            operations = []
            for idx in range(3, len(df)):
                row = df.iloc[idx]
                
                # 检查这一行是否有足够的列
                if len(row) < 4:
                    continue
                
                # 检查是否为空行
                if pd.isna(row[0]) and pd.isna(row[1]) and pd.isna(row[2]) and pd.isna(row[3]):
                    continue
                
                # 尝试读取每一列
                try:
                    setup_time = float(row[0]) if not pd.isna(row[0]) else 0.0
                    fixed_time = float(row[1]) if not pd.isna(row[1]) else 0.0
                    unit_time = float(row[2]) if not pd.isna(row[2]) else 0.0
                    machines_str = str(row[3]).strip() if not pd.isna(row[3]) else ""
                    
                    if not machines_str or machines_str == 'nan':
                        continue
                    
                    # 解析可选机器列表
                    available_machines = []
                    for m in machines_str.split(','):
                        m = m.strip()
                        if m and m.replace('.', '', 1).isdigit():
                            machine_id = int(float(m))
                            available_machines.append(machine_id)
                            # 更新最大机器编号
                            if machine_id > self.actual_max_machine:
                                self.actual_max_machine = machine_id
                    
                    if not available_machines:
                        continue
                    
                    operation = {
                        'setup_time': setup_time,
                        'fixed_time': fixed_time,
                        'unit_time': unit_time,
                        'available_machines': available_machines
                    }
                    operations.append(operation)
                    
                except Exception as e:
                    continue
            
            if not operations:
                print(f"  ❌ 没有读取到有效工序!")
                return None
            
            print(f"  ✓ 成功读取 {len(operations)} 道工序")
            
            return {
                'order_num': order_num,
                'part_num': part_num,
                'quantity': quantity,
                'operations': operations
            }
            
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def split_into_batches(self, quantity: int, max_batch_size: int = 10) -> List[int]:
        """将工件数量分解为批次"""
        batches = []
        remaining = quantity
        
        while remaining > 0:
            batch_size = min(remaining, max_batch_size)
            batches.append(batch_size)
            remaining -= batch_size
        
        return batches
    
    def calculate_processing_time(self, operation: Dict, batch_size: int, 
                                  include_setup: bool = True) -> float:
        """计算工序的加工时间"""
        setup = operation['setup_time'] if include_setup else 0
        total_time = (setup + 
                     operation['fixed_time'] + 
                     operation['unit_time'] * batch_size)
        return total_time
    
    def process_all_orders(self, max_batch_size: int = 40):
        """处理所有订单"""
        # 判断是文件还是文件夹
        if os.path.isfile(self.excel_folder):
            excel_files = [self.excel_folder]
            print(f"\n处理单个文件")
        elif os.path.isdir(self.excel_folder):
            excel_files = [
                os.path.join(self.excel_folder, f) 
                for f in os.listdir(self.excel_folder) 
                if f.endswith(('.xlsx', '.xls'))
            ]
            if not excel_files:
                print(f"❌ 文件夹中没有Excel文件: {self.excel_folder}")
                return
            print(f"\n找到 {len(excel_files)} 个Excel文件")
        else:
            print(f"❌ 路径不存在: {self.excel_folder}")
            return
        
        print("="*60)
        
        job_id = 0
        
        for filepath in sorted(excel_files):
            order = self.read_excel_file(filepath)
            
            if order is None or not order['operations']:
                print(f"  ⚠ 跳过此文件")
                continue
            
            # 将订单数量分解为批次
            batches = self.split_into_batches(order['quantity'], max_batch_size)
            print(f"  分解为 {len(batches)} 个批次: {batches}\n")
            
            # 为每个批次创建一个"工件"
            for batch_idx, batch_size in enumerate(batches):
                # 记录这个工件的工序数
                self.jobs_dict[job_id] = len(order['operations'])
                
                # 保存元数据
                self.job_metadata[job_id] = {
                    'order_num': order['order_num'],
                    'part_num': order['part_num'],
                    'batch_idx': batch_idx,
                    'batch_size': batch_size,
                    'total_batches': len(batches),
                    'operations': order['operations']
                }
                
                # 为每道工序生成加工时间行
                for op_idx, operation in enumerate(order['operations']):
                    # 含调机时间
                    proc_time_with_setup = self.calculate_processing_time(
                        operation, batch_size, include_setup=True
                    )
                    # 不含调机时间
                    proc_time_no_setup = self.calculate_processing_time(
                        operation, batch_size, include_setup=False
                    )
                    
                    time_row_with_setup = [9999] * self.max_machines
                    time_row_no_setup = [9999] * self.max_machines
                    
                    for machine_id in operation['available_machines']:
                        if 0 <= machine_id < self.max_machines:
                            time_row_with_setup[machine_id] = proc_time_with_setup
                            time_row_no_setup[machine_id] = proc_time_no_setup
                    
                    self.processing_time.append(time_row_with_setup)
                    self.processing_time_no_setup.append(time_row_no_setup)
                    self.total_operations += 1
                
                print(f"    工件{job_id}: 批次{batch_idx+1}, 数量={batch_size}, "
                      f"工序={len(order['operations'])}")
                
                job_id += 1
        
        # 调整max_machines为实际需要的大小(+1因为索引从0开始)
        if self.actual_max_machine > 0:
            self.max_machines = self.actual_max_machine + 1
            print(f"\n✓ 检测到最大机器编号: {self.actual_max_machine}")
            print(f"  自动调整机器数量为: {self.max_machines}")
        
        print("\n" + "="*60)
        print(f"✓ 处理完成!")
        print(f"  总工件数(批次数): {len(self.jobs_dict)}")
        print(f"  总工序数: {self.total_operations}")
        print(f"  机器数: {self.max_machines} (0-{self.max_machines-1})")
        print(f"  加工时间矩阵: {len(self.processing_time)}行 × {self.max_machines}列")
    
    def get_processing_time_without_setup(self, job_id: int, op_idx: int) -> Dict[int, float]:
        """获取指定工序不含调机时间的加工时间"""
        if job_id not in self.job_metadata:
            return {}
        
        metadata = self.job_metadata[job_id]
        if op_idx >= len(metadata['operations']):
            return {}
        
        operation = metadata['operations'][op_idx]
        batch_size = metadata['batch_size']
        
        proc_time = self.calculate_processing_time(operation, batch_size, 
                                                   include_setup=False)
        
        result = {}
        for machine_id in operation['available_machines']:
            if 0 <= machine_id < self.max_machines:
                result[machine_id] = proc_time
        
        return result
    
    def get_setup_time(self, job_id: int, op_idx: int) -> float:
        """获取指定工序的调机时间"""
        if job_id not in self.job_metadata:
            return 0
        
        metadata = self.job_metadata[job_id]
        if op_idx >= len(metadata['operations']):
            return 0
        
        return metadata['operations'][op_idx]['setup_time']
    
    def get_fjsp_data(self) -> Tuple[List[List[float]], Dict[int, int], int, int, int]:
        """获取FJSP求解所需的数据格式"""
        return (
            self.processing_time,
            self.jobs_dict,
            self.max_machines,
            self.total_operations,
            len(self.jobs_dict)
        )
    
    def export_to_excel(self, output_file='0_输入数据验证.xlsx'):
        """导出数据到Excel以便验证"""
        if not self.processing_time:
            print("❌ 没有数据可导出")
            return
        
        print(f"\n正在导出输入数据验证到Excel...")
        
        # 创建多个工作表
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 工作表1: 批次信息
            batch_data = []
            for job_id in sorted(self.job_metadata.keys()):
                meta = self.job_metadata[job_id]
                batch_data.append({
                    '工件ID': job_id,
                    '订单号': meta['order_num'],
                    '工件号': meta['part_num'],
                    '批次': f"{meta['batch_idx']+1}/{meta['total_batches']}",
                    '数量': meta['batch_size'],
                    '工序数': len(meta['operations'])
                })
            
            df_batch = pd.DataFrame(batch_data)
            df_batch.to_excel(writer, sheet_name='批次信息', index=False)
            
            # 工作表2: 工序详情
            operation_data = []
            for job_id in sorted(self.job_metadata.keys()):
                meta = self.job_metadata[job_id]
                for op_idx, operation in enumerate(meta['operations']):
                    proc_time = self.calculate_processing_time(operation, meta['batch_size'], True)
                    proc_time_no_setup = self.calculate_processing_time(operation, meta['batch_size'], False)
                    
                    operation_data.append({
                        '工件ID': job_id,
                        '订单-工件': f"{meta['order_num']}-{meta['part_num']}",
                        '批次': f"{meta['batch_idx']+1}",
                        '工序号': op_idx + 1,
                        '调机时间': operation['setup_time'],
                        '固定时间': operation['fixed_time'],
                        '单件时间': operation['unit_time'],
                        '批次数量': meta['batch_size'],
                        '总时间(含调机)': proc_time,
                        '总时间(不含调机)': proc_time_no_setup,
                        '可选机器': ','.join(map(str, operation['available_machines']))
                    })
            
            df_operations = pd.DataFrame(operation_data)
            df_operations.to_excel(writer, sheet_name='工序详情', index=False)
            
            # 工作表3: 机器使用统计
            machine_usage = {}
            for job_id in sorted(self.job_metadata.keys()):
                meta = self.job_metadata[job_id]
                for operation in meta['operations']:
                    for machine_id in operation['available_machines']:
                        if machine_id not in machine_usage:
                            machine_usage[machine_id] = 0
                        machine_usage[machine_id] += 1
            
            machine_data = [
                {'机器编号': m, '可用工序数': count}
                for m, count in sorted(machine_usage.items())
            ]
            df_machines = pd.DataFrame(machine_data)
            df_machines.to_excel(writer, sheet_name='机器使用统计', index=False)
            
            # 工作表4: 加工时间矩阵示例(前20行)
            matrix_data = []
            for i, row in enumerate(self.processing_time[:min(20, len(self.processing_time))]):
                row_dict = {'全局工序': i}
                for m, time in enumerate(row):
                    if time != 9999:
                        row_dict[f'M{m}'] = time
                matrix_data.append(row_dict)
            
            if matrix_data:
                df_matrix = pd.DataFrame(matrix_data)
                df_matrix.to_excel(writer, sheet_name='加工时间矩阵示例', index=False)
        
        print(f"✓ 输入数据验证已导出到: {output_file}")
        print(f"  包含4个工作表: 批次信息、工序详情、机器使用统计、加工时间矩阵示例")
    
    def print_summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("数据摘要:")
        print(f"  工件数(批次数): {len(self.jobs_dict)}")
        print(f"  总工序数: {self.total_operations}")
        print(f"  机器数: {self.max_machines} (编号0-{self.max_machines-1})")
        print(f"  实际最大机器编号: {self.actual_max_machine}")
        
        print("\n批次分组:")
        current_order = None
        for job_id, metadata in sorted(self.job_metadata.items()):
            order_part = f"{metadata['order_num']}-{metadata['part_num']}"
            if order_part != current_order:
                current_order = order_part
                print(f"\n  订单 {order_part}:")
            print(f"    工件{job_id}: 批次{metadata['batch_idx']+1}/{metadata['total_batches']}, "
                  f"数量={metadata['batch_size']}, 工序={len(metadata['operations'])}")
        print("="*60)


def main():
    """主函数"""
    processor = OrderProcessor(excel_folder='dingdan')
    processor.process_all_orders(max_batch_size=40)
    processor.print_summary()
    processor.export_to_excel('0_输入数据验证.xlsx')


if __name__ == '__main__':
    main()