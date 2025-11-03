"""
使用Google OR-Tools CP-SAT求解器求解柔性作业车间调度问题(FJSP)
优化目标:最小化makespan(最大完成时间)
"""

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from Instance import Processing_time, J, M_num, O_num, J_num

class FJSPSolver:
    def __init__(self):
        self.model = cp_model.CpModel()
        
        # 展平Processing_time为O_num x M_num矩阵
        # 原始格式: [job1的所有工序, job2的所有工序, ...]
        # 目标格式: [工序0, 工序1, ..., 工序O_num-1]
        self.processing_time_flat = []
        # print(len(Processing_time))

        """
        [
            [工序1-1],[工序1-2],...,[工序1-n],
            [工序2-1],[工序2-2],...,[工序2-n],
            ...
        ]
        """  
        for job_ops in Processing_time:
            for op_row in job_ops:
                self.processing_time_flat.append(op_row)
        
        print(f"展平后的Processing_time: {len(self.processing_time_flat)}行 x {M_num}列")
        
        # 从0开始
        # 处理J字典 - 转换为0-indexed
        if isinstance(J, dict):
            self.jobs = {}
            job_keys = sorted(J.keys())
            for idx, key in enumerate(job_keys):
                # 转化为从0开始,也可以修改instance.py中的J定义
                self.jobs[idx] = J[key]
            print(f"原始J字典: {J}")
            print(f"转换后jobs (0-indexed): {self.jobs}")
        else:
            self.jobs = {i: J[i] for i in range(len(J))}
        
        # 机器数 工件数 工序数
        self.num_machines = M_num
        self.num_jobs = J_num
        self.num_operations = O_num
        
        # 变量字典
        self.start_vars = {}      # (job, op) -> start time
        self.end_vars = {}        # (job, op) -> end time
        self.interval_vars = {}   # (job, op, machine) -> interval
        self.presence_vars = {}   # (job, op, machine) -> is selected
        
        self.makespan = None
        self.solver = cp_model.CpSolver()
        
    def build_model(self):
        """构建CP-SAT模型 - 遵循FJSP的9999约定和相对机器索引"""
        
        # 数据加载方式需要优化
        
        # 计算时间范围上限(horizon)
        horizon = 0
        for row in self.processing_time_flat:
            valid_times = [t for t in row if t != 9999]
            if valid_times:
                horizon += max(valid_times)
        
        horizon = int(horizon * 1.5)
        print(f"时间范围上限 (horizon): {horizon}")
        
        
        
        # 遍历所有工件和工序,构建CP-SAT变量
        operation_idx = 0  # 全局工序索引(对应processing_time_flat的行号)
        # 遍历每个工件
        """
            对每个工序设置变量和约束:
            - 可选区间变量: 每个工序在每台可用机器上的可选区间
            - 机器选择约束: 每个工序必须选择且仅选择一台机器
        """        
        for job_id in range(self.num_jobs):
            # 工序数
            num_ops = self.jobs[job_id]
            print(f"\n工件 J{job_id}: {num_ops}个工序")
            # 工序遍历
            for op_idx in range(num_ops):
                # 获取该工序的加工时间行
                proc_time_row = self.processing_time_flat[operation_idx]
                
                # 解析可用机器和加工时间
                available_machines = []
                processing_times = []
                for machine_id in range(self.num_machines):
                    proc_time = proc_time_row[machine_id]
                    if proc_time != 9999:
                        available_machines.append(machine_id)
                        processing_times.append(int(proc_time))
                
                # 如果没有可用的机器,则报错(数据有误)
                if not available_machines:
                    raise ValueError(
                        f"工件J{job_id}的工序O{op_idx}(全局索引{operation_idx})"
                        f"没有可用机器！\n加工时间行: {proc_time_row}"
                    )
                
                # 打印一下信息
                print(f"  J{job_id}-O{op_idx} (全局#{operation_idx}): "
                      f"可用机器{available_machines}, 加工时间{processing_times}")
                
                # 每个工序的变量列表(方便同时对多个变量设置约束)
                intervals_for_operation = []
                presences_for_operation = []
                machine_start_vars = []
                machine_end_vars = []
                
                # 遍历工序的可用机器和对应的时间去创建变量
                for machine_id, proc_time in zip(available_machines, processing_times):
                    
                    # 创建开始、结束时间变量
                    start_var = self.model.NewIntVar(
                        0, horizon, 
                        f'start_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    end_var = self.model.NewIntVar(
                        0, horizon, 
                        f'end_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    # 在哪个机器用的
                    presence = self.model.NewBoolVar(
                        f'presence_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    # 创建可选区间变量(只有presence=True时才占用机器时间)
                    interval = self.model.NewOptionalIntervalVar(
                        start_var, proc_time, end_var, presence,
                        f'interval_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    # 保存一下变量
                    self.interval_vars[(job_id, op_idx, machine_id)] = interval
                    self.presence_vars[(job_id, op_idx, machine_id)] = presence
                    
                    # 工序的变量列表
                    intervals_for_operation.append(interval)
                    presences_for_operation.append(presence)
                    machine_start_vars.append((start_var, presence))
                    machine_end_vars.append((end_var, presence))
                                
                # 约束1:每个工序必须恰好选择一台机器
                self.model.AddExactlyOne(presences_for_operation)
                
                # 创建实际的开始/结束时间变量
                # 使用OnlyEnforceIf实现条件赋值
                actual_start = self.model.NewIntVar(
                    0, horizon, 
                    f'actual_start_j{job_id}_o{op_idx}'
                )
                actual_end = self.model.NewIntVar(
                    0, horizon, 
                    f'actual_end_j{job_id}_o{op_idx}'
                )
                
                # 如果某台机器被选中,则actual时间 = 该机器的时间
                for start_var, presence in machine_start_vars:
                    self.model.Add(actual_start == start_var).OnlyEnforceIf(presence)
                
                for end_var, presence in machine_end_vars:
                    self.model.Add(actual_end == end_var).OnlyEnforceIf(presence)
                
                self.start_vars[(job_id, op_idx)] = actual_start
                self.end_vars[(job_id, op_idx)] = actual_end
                
                operation_idx += 1
        
        print("\n添加FJSP核心约束...")
        
        # 约束2:工序先后约束 - 同一工件的工序必须按顺序执行
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id] - 1):
                self.model.Add(
                    self.start_vars[(job_id, op_idx + 1)] >= 
                    self.end_vars[(job_id, op_idx)]
                )
        
        # 约束3:机器资源约束 - 同一机器上的工序不能时间重叠
        for machine_id in range(self.num_machines):
            intervals_on_machine = []
            for job_id in range(self.num_jobs):
                for op_idx in range(self.jobs[job_id]):
                    if (job_id, op_idx, machine_id) in self.interval_vars:
                        intervals_on_machine.append(
                            self.interval_vars[(job_id, op_idx, machine_id)]
                        )
            
            if intervals_on_machine:
                self.model.AddNoOverlap(intervals_on_machine)
        
        # 目标函数:最小化makespan(最大完成时间)
        self.makespan = self.model.NewIntVar(0, horizon, 'makespan')
        
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id]):
                self.model.Add(
                    self.makespan >= self.end_vars[(job_id, op_idx)]
                )
        
        self.model.Minimize(self.makespan)
        print(" 模型构建完成！")
        
    def solve(self, time_limit_seconds=300):
        # 参数设置
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.log_search_progress = False
        self.solver.parameters.num_search_workers = 8
        
        print("\n" + "="*80)
        print("开始求解FJSP问题...")
        print(f"问题规模: {self.num_jobs}个工件, {self.num_operations}个工序, "
              f"{self.num_machines}台机器")
        print("="*80)
        
        # 开始求解
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL:
            print(f"\n 找到最优解！")
        elif status == cp_model.FEASIBLE:
            print(f"\n 找到可行解(时间限制内未证明最优)")
        else:
            print(f"\n 未找到可行解！状态: {self.solver.StatusName()}")
            return None
        
        # 解的值
        optimal_makespan = self.solver.Value(self.makespan)
        print(f"\n 求解结果:")
        print(f"  - 最优Makespan: {optimal_makespan}")
        print(f"  - 求解时间: {self.solver.WallTime():.2f}秒")
        print(f"  - 分支数: {self.solver.NumBranches()}")
        print(f"  - 冲突数: {self.solver.NumConflicts()}")
        return self.extract_solution()
    
    def extract_solution(self):
        """提取调度方案"""
        schedule = []
        
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id]):
                start_time = self.solver.Value(self.start_vars[(job_id, op_idx)])
                end_time = self.solver.Value(self.end_vars[(job_id, op_idx)])
                
                # 找出被选中的机器
                selected_machine = None
                for machine_id in range(self.num_machines):
                    if (job_id, op_idx, machine_id) in self.presence_vars:
                        if self.solver.Value(self.presence_vars[(job_id, op_idx, machine_id)]):
                            selected_machine = machine_id
                            break
                
                schedule.append({
                    'job': job_id,
                    'operation': op_idx,
                    'machine': selected_machine,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
        
        return schedule
    
    def plot_gantt_chart(self, schedule, filename='ortools_optimal_schedule.png'):
        """绘制甘特图"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 为每个工件分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_jobs))
        
        # 按机器分组
        machines_schedule = {m: [] for m in range(self.num_machines)}
        for task in schedule:
            machines_schedule[task['machine']].append(task)
        
        # 绘制每台机器的任务
        for machine_id in range(self.num_machines):
            tasks = sorted(machines_schedule[machine_id], key=lambda x: x['start'])
            for task in tasks:
                job_id = task['job']
                start = task['start']
                duration = task['duration']
                
                # 绘制任务矩形
                rect = mpatches.Rectangle(
                    (start, machine_id - 0.4),
                    duration,
                    0.8,
                    facecolor=colors[job_id],
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.add_patch(rect)
                
                # 添加标签(工件-工序)
                text_color = 'white' if np.mean(colors[job_id][:3]) < 0.5 else 'black'
                ax.text(
                    start + duration / 2,
                    machine_id,
                    f'J{job_id}-O{task["operation"]}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color=text_color
                )
        
        # 设置坐标轴
        makespan = self.solver.Value(self.makespan)
        ax.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax.set_ylabel('机器', fontsize=13, fontweight='bold')
        ax.set_title(
            f'OR-Tools求解的FJSP最优调度方案 (Makespan={makespan})', 
            fontsize=15, fontweight='bold'
        )
        
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(self.num_machines)])
        ax.set_ylim(-0.5, self.num_machines - 0.5)
        ax.set_xlim(-2, makespan * 1.05)
        
        # 网格
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 图例
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f'工件 {i}') 
            for i in range(self.num_jobs)
        ]
        ax.legend(
            handles=legend_patches, 
            loc='upper right', 
            fontsize=10, 
            ncol=2 if self.num_jobs > 6 else 1
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n 甘特图已保存: {filename}")
        plt.close()
        
    def print_schedule(self, schedule):
        """打印详细调度方案"""
        print("\n" + "="*90)
        print("详细调度方案:")
        print("="*90)
        print(f"{'工件':<10}{'工序':<10}{'机器':<10}{'开始时间':<12}{'结束时间':<12}{'加工时间':<10}")
        print("-"*90)
        
        for task in sorted(schedule, key=lambda x: (x['job'], x['operation'])):
            print(f"J{task['job']:<9}O{task['operation']:<9}M{task['machine']:<9}"
                  f"{task['start']:<12}{task['end']:<12}{task['duration']:<10}")
        
        print("="*90)
        
        # 机器利用率分析
        print("\n机器利用率分析:")
        print("-"*50)
        makespan = self.solver.Value(self.makespan)
        for machine_id in range(self.num_machines):
            tasks_on_machine = [t for t in schedule if t['machine'] == machine_id]
            total_processing = sum(t['duration'] for t in tasks_on_machine)
            utilization = (total_processing / makespan * 100) if makespan > 0 else 0
            print(f"M{machine_id}: 总加工时间={total_processing:3d}, 利用率={utilization:5.2f}%")
        print("-"*50)