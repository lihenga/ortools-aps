"""
插单模块
从saved_schedules加载原有排产，从chadan文件夹读取新订单，进行插单求解
"""

from schedule_saver import ScheduleSaver, ScheduleLoader
from Pre import OrderProcessor
from tools import FJSPSolver
from ortools.sat.python import cp_model
from datetime import datetime
import os
import glob

class InsertOrderSolver:
    """插单求解器"""
    
    def __init__(self, base_schedule_file):
        """
        初始化插单求解器
        
        Args:
            base_schedule_file: 基础排产的pickle文件路径
        """
        # 加载基础排产
        print("="*80)
        print("步骤1: 加载基础排产")
        print("="*80)
        
        saver = ScheduleSaver()
        save_data = saver.load_schedule(base_schedule_file)
        self.loader = ScheduleLoader(save_data)
        
        # 提取基础数据
        self.base_schedule = self.loader.schedule
        self.base_makespan = self.loader.solver_data['makespan']
        self.num_machines = self.loader.solver_data['num_machines']
        
        # 构建机器时间线（哪些时间段被占用）
        self.base_machine_timeline = {}
        for task in self.base_schedule:
            machine = task['machine']
            if machine not in self.base_machine_timeline:
                self.base_machine_timeline[machine] = []
            
            self.base_machine_timeline[machine].append({
                'job': task['job'],
                'operation': task['operation'],
                'start': task['start'],
                'end': task['end']
            })
        
        # 按开始时间排序
        for machine in self.base_machine_timeline:
            self.base_machine_timeline[machine].sort(key=lambda x: x['start'])
        
        # 基础工件的数量
        self.base_num_jobs = self.loader.solver_data['num_jobs']
        
        print(f"✓ 基础排产加载成功")
        print(f"  当前Makespan: {self.base_makespan}")
        print(f"  工件数: {self.base_num_jobs}")
        print(f"  机器数: {self.num_machines}")
        
        # 打印每台机器的占用情况
        print(f"\n📊 机器占用情况:")
        for machine_id in sorted(self.base_machine_timeline.keys())[:5]:
            tasks = self.base_machine_timeline[machine_id]
            print(f"  机器M{machine_id}: {len(tasks)}个任务")
        if len(self.base_machine_timeline) > 5:
            print(f"  ... 还有 {len(self.base_machine_timeline) - 5} 台机器")
        
        # 新订单数据
        self.new_processor = None
        self.new_num_jobs = 0
        self.solver = None
    
    def load_new_orders(self, excel_folder='chadan', max_batch_size=40):
        """
        从chadan文件夹加载新订单
        
        Args:
            excel_folder: 新订单Excel文件夹
            max_batch_size: 最大批次大小
        """
        print("\n" + "="*80)
        print("步骤2: 加载新订单")
        print("="*80)
        
        if not os.path.exists(excel_folder):
            raise FileNotFoundError(f"插单文件夹不存在: {excel_folder}")
        
        # 检查文件夹中是否有Excel文件
        excel_files = glob.glob(os.path.join(excel_folder, '*.xlsx'))
        if not excel_files:
            raise FileNotFoundError(f"插单文件夹中没有Excel文件: {excel_folder}")
        
        print(f"✓ 找到 {len(excel_files)} 个Excel文件:")
        for f in excel_files:
            print(f"  - {os.path.basename(f)}")
        
        # 读取新订单
        self.new_processor = OrderProcessor(excel_folder=excel_folder)
        self.new_processor.process_all_orders(max_batch_size=max_batch_size)
        
        # 获取新订单的FJSP数据
        Processing_time, J, M_num, O_num, J_num = self.new_processor.get_fjsp_data()
        self.new_num_jobs = J_num
        
        print(f"\n✓ 新订单加载成功")
        print(f"  新增工件数: {self.new_num_jobs}")
        print(f"  新增工序数: {O_num}")
        
        self.new_processor.print_summary()
        
        # 导出新订单验证
        self.new_processor.export_to_excel('插单_新订单验证.xlsx')
        
        return self.new_num_jobs
    
    def build_insert_model(self):
        """
        构建插单模型
        核心思路: 原有工件的时间和机器固定，只对新工件进行优化
        """
        print("\n" + "="*80)
        print("步骤3: 构建插单模型")
        print("="*80)
        
        if self.new_processor is None:
            raise ValueError("请先调用 load_new_orders() 加载新订单")
        
        # 创建新的求解器实例（只对新订单建模）
        self.solver = FJSPSolver(processor=self.new_processor)
        
        # 使用原有的build_model
        print("构建基础模型...")
        self.solver.build_model()
        
        # 添加插单约束：新工件不能与原有工件在时间和机器上冲突
        print("\n添加插单约束...")
        self._add_no_conflict_constraints()
        
        print("✓ 插单模型构建完成")
    
    def _add_no_conflict_constraints(self):
        """
        添加无冲突约束：新工件的任务不能与原有工件的任务在同一机器上重叠
        """
        model = self.solver.model
        constraint_count = 0
        
        # 遍历所有机器
        for machine_id in range(self.num_machines):
            if machine_id not in self.base_machine_timeline:
                continue  # 这台机器在基础排产中没有任务
            
            base_tasks = self.base_machine_timeline[machine_id]
            
            # 遍历新工件的所有任务
            for new_job_id in range(self.new_num_jobs):
                if new_job_id not in self.solver.jobs:
                    continue
                
                num_ops = self.solver.jobs[new_job_id]
                
                for op_idx in range(num_ops):
                    # 检查这个新任务是否可以在这台机器上执行
                    if (new_job_id, op_idx, machine_id) not in self.solver.presence_vars:
                        continue
                    
                    presence = self.solver.presence_vars[(new_job_id, op_idx, machine_id)]
                    start_var = self.solver.start_vars[(new_job_id, op_idx)]
                    end_var = self.solver.end_vars[(new_job_id, op_idx)]
                    
                    # 对于每个已存在的任务，新任务必须在它之前或之后
                    for base_task in base_tasks:
                        base_start = int(base_task['start'])
                        base_end = int(base_task['end'])
                        
                        # 创建布尔变量：新任务在基础任务之前
                        before = model.NewBoolVar(
                            f'new_j{new_job_id}_o{op_idx}_before_base_j{base_task["job"]}_m{machine_id}'
                        )
                        
                        # 创建布尔变量：新任务在基础任务之后
                        after = model.NewBoolVar(
                            f'new_j{new_job_id}_o{op_idx}_after_base_j{base_task["job"]}_m{machine_id}'
                        )
                        
                        # 如果新任务在这台机器上，则必须满足：在之前 OR 在之后
                        # before => end_var <= base_start
                        model.Add(end_var <= base_start).OnlyEnforceIf([presence, before])
                        
                        # after => start_var >= base_end
                        model.Add(start_var >= base_end).OnlyEnforceIf([presence, after])
                        
                        # 必须二选一（如果新任务在这台机器上）
                        model.AddBoolOr([before, after]).OnlyEnforceIf(presence)
                        
                        constraint_count += 1
            
            if constraint_count % 100 == 0 and constraint_count > 0:
                print(f"  已添加 {constraint_count} 个冲突约束...")
        
        print(f"✓ 插单约束添加完成 (共{constraint_count}个约束)")
    
    def solve(self, time_limit_seconds=300):
        """
        求解插单问题
        
        Args:
            time_limit_seconds: 求解时间限制
        
        Returns:
            新订单的调度方案, 新的makespan
        """
        print("\n" + "="*80)
        print("步骤4: 求解插单问题")
        print("="*80)
        
        if self.solver is None:
            raise ValueError("请先调用 build_insert_model() 构建模型")
        
        # 求解（只求解新订单）
        new_schedule = self.solver.solve(time_limit_seconds=time_limit_seconds)
        
        if new_schedule:
            # 合并方案：基础方案 + 新方案
            combined_schedule = self._combine_schedules(new_schedule)
            
            # 计算新的makespan
            new_makespan = max(task['end'] for task in combined_schedule)
            
            print(f"\n✓ 插单完成!")
            print(f"  原Makespan: {self.base_makespan}")
            print(f"  新Makespan: {new_makespan}")
            print(f"  增加时间: {new_makespan - self.base_makespan} (+{(new_makespan - self.base_makespan) / self.base_makespan * 100:.2f}%)")
            
            return combined_schedule, new_makespan
        else:
            print("\n❌ 插单求解失败")
            return None, None
    
    def _combine_schedules(self, new_schedule):
        """
        合并基础排产和新订单排产
        
        Args:
            new_schedule: 新订单的调度方案
        
        Returns:
            合并后的完整调度方案
        """
        combined = []
        
        # 1. 添加基础排产（保持不变）
        for task in self.base_schedule:
            combined.append({
                'job': task['job'],
                'operation': task['operation'],
                'machine': task['machine'],
                'start': task['start'],
                'end': task['end'],
                'duration': task['duration'],
                'is_new': False  # 标记为原有订单
            })
        
        # 2. 添加新订单排产（job编号需要偏移）
        for task in new_schedule:
            combined.append({
                'job': self.base_num_jobs + task['job'],  # 偏移job_id避免冲突
                'operation': task['operation'],
                'machine': task['machine'],
                'start': task['start'],
                'end': task['end'],
                'duration': task['duration'],
                'is_new': True  # 标记为新订单
            })
        
        return combined
    
    def export_result(self, combined_schedule, new_makespan):
        """
        导出插单结果
        
        Args:
            combined_schedule: 合并后的调度方案
            new_makespan: 新的makespan
        """
        print("\n" + "="*80)
        print("步骤5: 导出插单结果")
        print("="*80)
        
        # 创建new_saved_schedules文件夹（如果不存在）
        output_folder = 'new_saved_schedules'
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出Excel
        excel_filename = f'{output_folder}/插单结果_{timestamp}.xlsx'
        self._export_to_excel(combined_schedule, new_makespan, excel_filename)
        
        # 2. 生成甘特图
        gantt_filename = f'{output_folder}/插单甘特图_{timestamp}.png'
        self._plot_gantt(combined_schedule, new_makespan, gantt_filename)
        
        # 3. 保存完整的插单结果（用于后续再次插单）
        pickle_filename = f'{output_folder}/插单结果_{timestamp}.pkl'
        self._save_combined_schedule(combined_schedule, new_makespan, pickle_filename)
        
        print(f"\n✓ 插单结果导出完成:")
        print(f"  Excel: {excel_filename}")
        print(f"  甘特图: {gantt_filename}")
        print(f"  Pickle: {pickle_filename}")
        print(f"\n  可将 {pickle_filename} 移至 saved_schedules 文件夹后用于下次插单")

    def _export_to_excel(self, combined_schedule, new_makespan, filename):
        """导出Excel"""
        import pandas as pd
        
        # 构建DataFrame
        df_data = []
        for task in sorted(combined_schedule, key=lambda x: (x['machine'], x['start'])):
            df_data.append({
                '工件ID': task['job'],
                '工序': task['operation'] + 1,
                '机器': task['machine'],
                '开始时间': task['start'],
                '结束时间': task['end'],
                '时长': task['duration'],
                '类型': '新插单' if task['is_new'] else '原有订单'
            })
        
        df = pd.DataFrame(df_data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='插单结果', index=False)
            
            # 按机器分组
            df_by_machine = df.groupby('机器').agg({
                '时长': 'sum',
                '工件ID': 'count'
            }).rename(columns={'工件ID': '任务数'})
            df_by_machine['利用率(%)'] = (df_by_machine['时长'] / new_makespan) * 100
            df_by_machine.to_excel(writer, sheet_name='机器利用率')
            
            # 新旧订单统计
            df_summary = df.groupby('类型').agg({
                '工件ID': 'nunique',
                '时长': 'sum'
            }).rename(columns={'工件ID': '工件数', '时长': '总时长'})
            df_summary.to_excel(writer, sheet_name='订单统计')
        
        print(f"  ✓ Excel已保存: {filename}")
    
    def _plot_gantt(self, combined_schedule, new_makespan, filename):
        """生成甘特图（支持热处理批处理的上下并排显示）"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 按机器分组
        machine_tasks = {i: [] for i in range(self.num_machines)}
        for task in combined_schedule:
            machine_tasks[task['machine']].append(task)
        
        # 计算总工件数
        all_job_ids = sorted(set(task['job'] for task in combined_schedule))
        num_jobs = len(all_job_ids)
        
        # 颜色方案
        colors_map = {}
        color_palette = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        for idx, job_id in enumerate(all_job_ids):
            colors_map[job_id] = color_palette[idx]
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # 热处理机器列表
        HEAT_TREATMENT_MACHINES = [10, 11, 12]
        
        # 绘制每台机器的任务
        for machine_id in range(self.num_machines):
            tasks = sorted(machine_tasks[machine_id], key=lambda x: (x['start'], x['job']))
            
            if machine_id in HEAT_TREATMENT_MACHINES and tasks:
                # ✅ 热处理机器：批处理显示逻辑（上下并排）
                batch_groups = []
                processed = set()
                
                # 找出批处理组（开始和结束时间完全相同的任务）
                for i, task_i in enumerate(tasks):
                    if i in processed:
                        continue
                    batch = [task_i]
                    for j, task_j in enumerate(tasks):
                        if j != i and j not in processed:
                            if (task_i['start'] == task_j['start'] and 
                                task_i['end'] == task_j['end']):
                                batch.append(task_j)
                                processed.add(j)
                    processed.add(i)
                    batch_groups.append(batch)
                
                # 绘制批处理组
                for batch_idx, batch in enumerate(batch_groups):
                    if len(batch) > 1:
                        # ✅ 多个任务批处理：上下分割显示
                        total_height = 0.8
                        sub_height = total_height / len(batch)
                        
                        for sub_idx, task in enumerate(batch):
                            y_pos = machine_id - 0.4 + sub_idx * sub_height
                            color = colors_map[task['job']]
                            duration = task['end'] - task['start']
                            
                            # 根据是否为新插单调整边框
                            edge_color = 'red' if task['is_new'] else 'black'
                            edge_width = 2.5 if task['is_new'] else 1.2
                            
                            rect = mpatches.Rectangle(
                                (task['start'], y_pos),
                                duration,
                                sub_height * 0.95,
                                facecolor=color,
                                edgecolor=edge_color,
                                linewidth=edge_width,
                                alpha=0.85
                            )
                            ax.add_patch(rect)
                            
                            # 文本标签
                            prefix = '[新]' if task['is_new'] else ''
                            text = f"{prefix}J{task['job']}-O{task['operation']}"
                            ax.text(
                                task['start'] + duration / 2,
                                y_pos + sub_height * 0.475,
                                text,
                                ha='center',
                                va='center',
                                fontsize=max(7, 10 - len(batch) // 2),
                                fontweight='bold',
                                color='white'
                            )
                        
                        # ✅ 红色虚线框标记批处理（批处理 + 新插单标记）
                        has_new = any(task['is_new'] for task in batch)
                        batch_rect = mpatches.Rectangle(
                            (batch[0]['start'], machine_id - 0.4),
                            batch[0]['end'] - batch[0]['start'],
                            total_height,
                            facecolor='none',
                            edgecolor='red' if has_new else 'orange',
                            linewidth=3.5 if has_new else 3,
                            linestyle='--',
                            alpha=0.9
                        )
                        ax.add_patch(batch_rect)
                    else:
                        # ✅ 单个任务
                        task = batch[0]
                        y_pos = machine_id - 0.4
                        color = colors_map[task['job']]
                        duration = task['end'] - task['start']
                        
                        # 根据是否为新插单调整边框
                        edge_color = 'red' if task['is_new'] else 'black'
                        edge_width = 2.5 if task['is_new'] else 1.5
                        
                        rect = mpatches.Rectangle(
                            (task['start'], y_pos),
                            duration,
                            0.8,
                            facecolor=color,
                            edgecolor=edge_color,
                            linewidth=edge_width,
                            alpha=0.85
                        )
                        ax.add_patch(rect)
                        
                        prefix = '[新]' if task['is_new'] else ''
                        text = f"{prefix}J{task['job']}-O{task['operation']}"
                        ax.text(
                            task['start'] + duration / 2,
                            machine_id,
                            text,
                            ha='center',
                            va='center',
                            fontsize=10,
                            fontweight='bold',
                            color='white'
                        )
            else:
                # ✅ 普通机器：标准显示
                for task in tasks:
                    color = colors_map[task['job']]
                    
                    # 根据是否为新插单调整边框
                    edge_color = 'red' if task['is_new'] else 'black'
                    edge_width = 2.5 if task['is_new'] else 1.5
                    
                    rect = mpatches.Rectangle(
                        (task['start'], machine_id - 0.4),
                        task['duration'],
                        0.8,
                        facecolor=color,
                        edgecolor=edge_color,
                        linewidth=edge_width,
                        alpha=0.85
                    )
                    ax.add_patch(rect)
                    
                    prefix = '[新]' if task['is_new'] else ''
                    text = f"{prefix}J{task['job']}-O{task['operation']}"
                    ax.text(
                        task['start'] + task['duration'] / 2,
                        machine_id,
                        text,
                        ha='center',
                        va='center',
                        fontsize=10,
                        fontweight='bold',
                        color='white'
                    )
    
        # ✅ 设置坐标轴（注意缩进，必须在 for 循环外）
        ax.set_xlabel('时间', fontsize=14, fontweight='bold')
        ax.set_ylabel('机器', fontsize=14, fontweight='bold')
        
        # 标题
        increase = new_makespan - self.base_makespan
        increase_pct = (increase / self.base_makespan) * 100
        title = (f'插单后的排产方案 (Makespan={new_makespan:.0f})\n'
                 f'原Makespan: {self.base_makespan:.0f} → 新Makespan: {new_makespan:.0f} '
                 f'(+{increase:.0f}分钟, +{increase_pct:.1f}%)\n'
                 f'✓ 红色边框 = 新插单 | 红色虚线框 = 批处理(含新插单) | 橙色虚线框 = 批处理(原有订单)')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Y轴标签
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([
            f'M{i} (热处理)' if i in HEAT_TREATMENT_MACHINES else f'M{i}' 
            for i in range(self.num_machines)
        ], fontsize=11)
        ax.set_ylim(-0.5, self.num_machines - 0.5)
        ax.set_xlim(-15, new_makespan * 1.05)
        
        # 网格
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 图例
        legend_patches = []
        for job_id in all_job_ids:
            is_new_job = any(task['is_new'] and task['job'] == job_id for task in combined_schedule)
            label = f"工件 {job_id} {'[新]' if is_new_job else ''}"
            
            legend_patches.append(
                mpatches.Patch(
                    color=colors_map[job_id], 
                    label=label, 
                    alpha=0.85,
                    edgecolor='red' if is_new_job else 'black',
                    linewidth=2 if is_new_job else 1
                )
            )
        
        legend_patches.append(
            mpatches.Patch(
                facecolor='gray', 
                edgecolor='red', 
                linewidth=2.5, 
                label='✓ 新插单任务'
            )
        )
        legend_patches.append(
            mpatches.Patch(
                facecolor='none', 
                edgecolor='red', 
                linewidth=3.5, 
                linestyle='--',
                label='✓ 批处理(含新插单)'
            )
        )
        legend_patches.append(
            mpatches.Patch(
                facecolor='none', 
                edgecolor='orange', 
                linewidth=3, 
                linestyle='--',
                label='✓ 批处理(原有订单)'
            )
        )
        
        ax.legend(
            handles=legend_patches, 
            loc='upper right', 
            fontsize=10, 
            ncol=2,
            framealpha=0.95,
            edgecolor='black'
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ 甘特图已保存: {filename}")
        plt.close()
    
    def _save_combined_schedule(self, combined_schedule, new_makespan, pickle_filename):
        """保存合并后的完整排产（用于后续再次插单）"""
        import pickle
        
        # 重建完整的save_data结构
        save_data = {
            'schedule': combined_schedule,
            'solver_data': {
                'num_jobs': len(set(task['job'] for task in combined_schedule)),
                'num_machines': self.num_machines,
                'num_operations': len(combined_schedule),
                'makespan': new_makespan,
                'status': 'INSERTION_COMPLETED',
                'horizon': int(new_makespan * 1.5),
            },
            'processor_data': None,
            'optimization_results': {
                'setup_optimization': [],
                'batch_processing': [],
                'total_setup_saved': 0,
            },
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(pickle_filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"  ✓ 完整排产已保存: {pickle_filename}")


def main():
    """主函数"""
    
    # 1. 列出所有可用的排产结果
    print("="*80)
    print("插单模块")
    print("="*80)
    
    saver = ScheduleSaver()
    save_dir = saver.save_dir
    
    if not os.path.exists(save_dir):
        print(f"\n❌ 未找到保存目录: {save_dir}")
        print("请先运行 toolsmain.py 生成排产结果")
        return
    
    # 列出所有pickle文件
    pickle_files = glob.glob(os.path.join(save_dir, '*.pkl'))
    
    if not pickle_files:
        print(f"\n❌ 在 {save_dir} 目录下未找到排产结果文件")
        print("请先运行 toolsmain.py 生成排产结果")
        return
    
    pickle_files.sort(reverse=True)
    
    print(f"\n找到 {len(pickle_files)} 个已保存的排产结果:\n")
    
    schedules = []
    for i, filepath in enumerate(pickle_files, 1):
        try:
            save_data = saver.load_schedule(filepath)
            loader = ScheduleLoader(save_data)
            
            schedules.append({
                'index': i,
                'file': filepath,
                'name': os.path.basename(filepath),
                'makespan': loader.solver_data['makespan'],
                'num_jobs': loader.solver_data['num_jobs'],
                'timestamp': save_data['timestamp']
            })
            
            print(f"{i}. {os.path.basename(filepath)}")
            print(f"   Makespan: {loader.solver_data['makespan']}")
            print(f"   工件数: {loader.solver_data['num_jobs']}")
            print(f"   时间戳: {save_data['timestamp']}")
            print()
        except Exception as e:
            print(f"{i}. {os.path.basename(filepath)} (加载失败: {e})")
    
    if not schedules:
        print("❌ 没有可用的排产结果")
        return
    
    # 2. 选择基础排产
    choice = input(f"\n请选择基础排产 (1-{len(schedules)}, 默认1): ").strip()
    idx = int(choice) - 1 if choice and choice.isdigit() else 0
    
    if idx < 0 or idx >= len(schedules):
        print("❌ 无效的选择")
        return
    
    base_schedule_file = schedules[idx]['file']
    print(f"\n✓ 已选择: {schedules[idx]['name']}")
    
    # 3. 创建插单求解器
    insert_solver = InsertOrderSolver(base_schedule_file)
    
    # 4. 加载新订单
    try:
        insert_solver.load_new_orders(excel_folder='chadan', max_batch_size=40)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("请确保 chadan 文件夹存在并包含新订单的Excel文件")
        return
    
    # 5. 构建插单模型
    insert_solver.build_insert_model()
    
    # 6. 求解
    combined_schedule, new_makespan = insert_solver.solve(time_limit_seconds=60)
    
    if combined_schedule:
        # 7. 导出结果
        insert_solver.export_result(combined_schedule, new_makespan)
        
        print("\n" + "="*80)
        print("✓ 插单完成!")
        print("="*80)
    else:
        print("\n❌ 插单失败，未找到可行解")
        print("可能的原因:")
        print("  1. 新订单的工序与现有排产冲突严重")
        print("  2. 求解时间不足，可尝试增加 time_limit_seconds")
        print("  3. 新订单的约束过于严格")


if __name__ == '__main__':
    main()