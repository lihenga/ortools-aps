"""
使用Google OR-Tools CP-SAT求解器求解柔性作业车间调度问题(FJSP)
优化目标:最小化makespan(最大完成时间) + 最大化调机时间节省
"""

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class FJSPSolver:
    def __init__(self, processor=None):
        """
        初始化求解器
        
        Args:
            processor: OrderProcessor实例,如果为None则从Instance.py读取
        """
        self.model = cp_model.CpModel()
        self.processor = processor
        
        if processor is not None:
            # 从Pre.py的OrderProcessor读取数据
            Processing_time, J, M_num, O_num, J_num = processor.get_fjsp_data()
            self.processing_time_flat = Processing_time
            self.processing_time_no_setup_flat = processor.processing_time_no_setup
            self.jobs = {i: J[i] for i in range(J_num)}
            self.num_machines = M_num
            self.num_jobs = J_num
            self.num_operations = O_num
            print(f"从OrderProcessor读取数据: {J_num}个工件, {O_num}个工序, {M_num}台机器")
        else:
            # 从Instance.py读取(保持兼容性)
            from Instance import Processing_time, J, M_num, O_num, J_num
            self.processing_time_flat = []
            for job_ops in Processing_time:
                for op_row in job_ops:
                    self.processing_time_flat.append(op_row)
            
            self.processing_time_no_setup_flat = self.processing_time_flat  # 兼容模式
            
            if isinstance(J, dict):
                self.jobs = {}
                job_keys = sorted(J.keys())
                for idx, key in enumerate(job_keys):
                    self.jobs[idx] = J[key]
            else:
                self.jobs = {i: J[i] for i in range(len(J))}
            
            self.num_machines = M_num
            self.num_jobs = J_num
            self.num_operations = O_num
            print(f"从Instance.py读取数据: {J_num}个工件, {O_num}个工序, {M_num}台机器")
        
        # 变量字典
        self.start_vars = {}
        self.end_vars = {}
        self.interval_vars = {}
        self.presence_vars = {}
        self.duration_vars = {}
        
        # 优化记录
        self.batch_processing_pairs = []  # 批处理对
        self.setup_optimization_pairs = []  # 调机优化对
        self.setup_saved_vars = []  # 记录所有调机节省的布尔变量
        
        self.makespan = None
        self.solver = cp_model.CpSolver()
        
    def build_model(self):
        """构建CP-SAT模型"""
        
        horizon = 0
        for row in self.processing_time_flat:
            valid_times = [t for t in row if t != 9999]
            if valid_times:
                horizon += max(valid_times)
        
        horizon = int(horizon * 1.5)
        print(f"时间范围上限: {horizon}")
        
        operation_idx = 0
        
        # 第1步: 为每个任务的每个可选机器创建变量
        for job_id in range(self.num_jobs):
            num_ops = self.jobs[job_id]
            
            for op_idx in range(num_ops):
                proc_time_row_with_setup = self.processing_time_flat[operation_idx]
                proc_time_row_no_setup = self.processing_time_no_setup_flat[operation_idx]
                
                available_machines = []
                processing_times_with_setup = []
                processing_times_no_setup = []
                
                for machine_id in range(self.num_machines):
                    proc_time_with = proc_time_row_with_setup[machine_id]
                    proc_time_without = proc_time_row_no_setup[machine_id]
                    
                    if proc_time_with != 9999:
                        available_machines.append(machine_id)
                        processing_times_with_setup.append(int(proc_time_with))
                        processing_times_no_setup.append(int(proc_time_without))
                
                if not available_machines:
                    raise ValueError(f"工件J{job_id}的工序O{op_idx}没有可用机器！")
                
                intervals_for_operation = []
                presences_for_operation = []
                machine_start_vars = []
                machine_end_vars = []
                
                for idx, machine_id in enumerate(available_machines):
                    proc_time_with = processing_times_with_setup[idx]
                    proc_time_without = processing_times_no_setup[idx]
                    
                    start_var = self.model.NewIntVar(
                        0, horizon, f'start_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    end_var = self.model.NewIntVar(
                        0, horizon, f'end_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    presence = self.model.NewBoolVar(
                        f'presence_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    # 创建可变的duration
                    duration_var = self.model.NewIntVar(
                        proc_time_without, proc_time_with, 
                        f'duration_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    # 建立关系: end = start + duration
                    self.model.Add(end_var == start_var + duration_var).OnlyEnforceIf(presence)
                    
                    # 使用可变duration的interval
                    interval = self.model.NewOptionalIntervalVar(
                        start_var, duration_var, end_var, presence,
                        f'interval_j{job_id}_o{op_idx}_m{machine_id}'
                    )
                    
                    self.interval_vars[(job_id, op_idx, machine_id)] = interval
                    self.presence_vars[(job_id, op_idx, machine_id)] = presence
                    
                    # 保存duration_var用于后续调机优化
                    self.duration_vars[(job_id, op_idx, machine_id)] = (
                        duration_var, proc_time_with, proc_time_without
                    )
                    
                    intervals_for_operation.append(interval)
                    presences_for_operation.append(presence)
                    machine_start_vars.append((start_var, presence))
                    machine_end_vars.append((end_var, presence))
                
                self.model.AddExactlyOne(presences_for_operation)
                
                actual_start = self.model.NewIntVar(0, horizon, f'actual_start_j{job_id}_o{op_idx}')
                actual_end = self.model.NewIntVar(0, horizon, f'actual_end_j{job_id}_o{op_idx}')
                
                for start_var, presence in machine_start_vars:
                    self.model.Add(actual_start == start_var).OnlyEnforceIf(presence)
                for end_var, presence in machine_end_vars:
                    self.model.Add(actual_end == end_var).OnlyEnforceIf(presence)
                
                self.start_vars[(job_id, op_idx)] = actual_start
                self.end_vars[(job_id, op_idx)] = actual_end
                
                operation_idx += 1
        
        # 第2步: 工序先后约束
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id] - 1):
                self.model.Add(
                    self.start_vars[(job_id, op_idx + 1)] >= 
                    self.end_vars[(job_id, op_idx)]
                )
        
        # 第3步: 机器资源约束 + 热处理批处理
        HEAT_TREATMENT_MACHINES = [15, 16]  # 热处理机器
        
        for machine_id in range(self.num_machines):
            intervals_on_machine = []
            tasks_on_machine = []
            
            for job_id in range(self.num_jobs):
                for op_idx in range(self.jobs[job_id]):
                    if (job_id, op_idx, machine_id) in self.interval_vars:
                        intervals_on_machine.append(self.interval_vars[(job_id, op_idx, machine_id)])
                        tasks_on_machine.append((job_id, op_idx))
            
            if intervals_on_machine:
                if machine_id in HEAT_TREATMENT_MACHINES:
                    # 热处理机器: 支持批处理
                    print(f"\n热处理机器 M{machine_id}: 添加批处理约束 (共{len(tasks_on_machine)}个任务)")
                    
                    for i in range(len(tasks_on_machine)):
                        for j in range(i + 1, len(tasks_on_machine)):
                            job_i, op_i = tasks_on_machine[i]
                            job_j, op_j = tasks_on_machine[j]
                            
                            start_i = self.start_vars[(job_i, op_i)]
                            end_i = self.end_vars[(job_i, op_i)]
                            start_j = self.start_vars[(job_j, op_j)]
                            end_j = self.end_vars[(job_j, op_j)]
                            
                            presence_i = self.presence_vars[(job_i, op_i, machine_id)]
                            presence_j = self.presence_vars[(job_j, op_j, machine_id)]
                            
                            both_present = self.model.NewBoolVar(
                                f'both_present_m{machine_id}_j{job_i}o{op_i}_j{job_j}o{op_j}'
                            )
                            
                            self.model.AddBoolAnd([presence_i, presence_j]).OnlyEnforceIf(both_present)
                            self.model.AddBoolOr([presence_i.Not(), presence_j.Not()]).OnlyEnforceIf(both_present.Not())
                            
                            fully_overlap = self.model.NewBoolVar(
                                f'fully_overlap_m{machine_id}_j{job_i}o{op_i}_j{job_j}o{op_j}'
                            )
                            i_before_j = self.model.NewBoolVar(
                                f'i_before_j_m{machine_id}_j{job_i}o{op_i}_j{job_j}o{op_j}'
                            )
                            j_before_i = self.model.NewBoolVar(
                                f'j_before_i_m{machine_id}_j{job_i}o{op_idx}_j{job_j}o{op_j}'
                            )
                            
                            self.model.Add(start_i == start_j).OnlyEnforceIf([both_present, fully_overlap])
                            self.model.Add(end_i == end_j).OnlyEnforceIf([both_present, fully_overlap])
                            self.model.Add(end_i <= start_j).OnlyEnforceIf([both_present, i_before_j])
                            self.model.Add(end_j <= start_i).OnlyEnforceIf([both_present, j_before_i])
                            
                            self.model.AddBoolOr([fully_overlap, i_before_j, j_before_i]).OnlyEnforceIf(both_present)
                            self.model.Add(fully_overlap + i_before_j + j_before_i == 1).OnlyEnforceIf(both_present)
                            
                            # 记录潜在的批处理对
                            self.batch_processing_pairs.append({
                                'machine': machine_id,
                                'task_i': (job_i, op_i),
                                'task_j': (job_j, op_j),
                                'overlap_var': fully_overlap
                            })
                else:
                    # 其他机器: 标准约束
                    self.model.AddNoOverlap(intervals_on_machine)
        
        # 第4步: 调机优化约束 (核心!)
        if self.processor is not None:
            self._add_setup_optimization_constraints()
        
        # 第5步: 多目标优化 - 主目标makespan + 次目标调机节省
        self.makespan = self.model.NewIntVar(0, horizon, 'makespan')
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id]):
                self.model.Add(self.makespan >= self.end_vars[(job_id, op_idx)])
        
        # 计算总的调机节省量
        if self.setup_saved_vars:
            total_setup_saved = sum(self.setup_saved_vars)
            # 多目标: 最小化makespan,同时最大化调机节省
            # 使用加权和: makespan - setup_saved * weight
            # weight设置为1,使得每节省1分钟调机时间等价于减少1分钟makespan
            self.model.Minimize(self.makespan - total_setup_saved)
            print(f"✓ 添加多目标优化: 最小化makespan, 最大化调机节省")
        else:
            self.model.Minimize(self.makespan)
        
        print("✓ 模型构建完成!")
    
    def _add_setup_optimization_constraints(self):
        """
        添加调机优化约束
        
        关键改进: 不仅添加"可以省略调机"的约束,还要激励求解器选择这种方案
        """
        
        print("\n添加调机优化约束...")
        
        # 按(订单号, 工件号, 工序号, 机器)分组所有任务
        task_groups = {}
        
        for job_id in range(self.num_jobs):
            meta = self.processor.job_metadata.get(job_id)
            if meta is None:
                continue
            
            for op_idx in range(self.jobs[job_id]):
                # 获取这个工序的调机时间
                setup_time = self.processor.get_setup_time(job_id, op_idx)
                if setup_time <= 0:
                    continue  # 没有调机时间,跳过
                
                # 获取可用机器
                for machine_id in range(self.num_machines):
                    if (job_id, op_idx, machine_id) not in self.presence_vars:
                        continue
                    
                    # 创建分组key: (订单号, 工件号, 工序号, 机器)
                    key = (meta['order_num'], meta['part_num'], op_idx, machine_id)
                    
                    if key not in task_groups:
                        task_groups[key] = []
                    
                    task_groups[key].append({
                        'job_id': job_id,
                        'op_idx': op_idx,
                        'machine_id': machine_id,
                        'batch_idx': meta['batch_idx'],
                        'setup_time': setup_time
                    })
        
        # 对每组添加约束
        setup_count = 0
        for key, tasks in task_groups.items():
            if len(tasks) < 2:
                continue  # 只有一个批次,不需要优化
            
            order_num, part_num, op_idx, machine_id = key
            
            # 按批次索引排序
            tasks = sorted(tasks, key=lambda x: x['batch_idx'])
            
            print(f"  订单{order_num}-工件{part_num}-工序{op_idx+1}-机器{machine_id}: "
                  f"发现{len(tasks)}个批次可优化")
            
            # 对每对可能连续的批次添加约束
            for i in range(len(tasks)):
                task_i = tasks[i]
                job_i = task_i['job_id']
                
                for j in range(len(tasks)):
                    if i == j:
                        continue
                    
                    task_j = tasks[j]
                    job_j = task_j['job_id']
                    
                    # 获取变量
                    presence_i = self.presence_vars[(job_i, op_idx, machine_id)]
                    presence_j = self.presence_vars[(job_j, op_idx, machine_id)]
                    
                    duration_j, time_with, time_without = self.duration_vars[(job_j, op_idx, machine_id)]
                    setup_time_value = time_with - time_without
                    
                    if setup_time_value <= 0:
                        continue
                    
                    end_i = self.end_vars[(job_i, op_idx)]
                    start_j = self.start_vars[(job_j, op_idx)]
                    
                    # 创建"j紧跟i"的布尔变量
                    j_follows_i = self.model.NewBoolVar(
                        f'follows_m{machine_id}_j{job_i}_to_j{job_j}_op{op_idx}'
                    )
                    
                    # 两者都在这台机器上
                    both_on_machine = self.model.NewBoolVar(
                        f'both_m{machine_id}_j{job_i}_j{job_j}_op{op_idx}'
                    )
                    self.model.AddBoolAnd([presence_i, presence_j]).OnlyEnforceIf(both_on_machine)
                    self.model.AddBoolOr([presence_i.Not(), presence_j.Not()]).OnlyEnforceIf(both_on_machine.Not())
                    
                    # j_follows_i => (both_on_machine AND end_i == start_j)
                    self.model.AddImplication(j_follows_i, both_on_machine)
                    self.model.Add(end_i == start_j).OnlyEnforceIf(j_follows_i)
                    
                    # 如果j_follows_i,则duration_j使用不含调机的时间
                    self.model.Add(duration_j == time_without).OnlyEnforceIf(j_follows_i)
                    
                    # 如果不紧跟且都在这台机器,则必须使用含调机的时间
                    not_follows_but_both = self.model.NewBoolVar(
                        f'not_follows_but_both_m{machine_id}_j{job_i}_j{job_j}_op{op_idx}'
                    )
                    self.model.AddBoolAnd([both_on_machine, j_follows_i.Not()]).OnlyEnforceIf(not_follows_but_both)
                    self.model.Add(duration_j == time_with).OnlyEnforceIf(not_follows_but_both)
                    
                    # 创建"节省调机时间"的整数变量,用于目标函数
                    setup_saved = self.model.NewIntVar(
                        0, setup_time_value,
                        f'setup_saved_m{machine_id}_j{job_i}_to_j{job_j}_op{op_idx}'
                    )
                    
                    # 如果j_follows_i,则setup_saved = setup_time_value,否则=0
                    self.model.Add(setup_saved == setup_time_value).OnlyEnforceIf(j_follows_i)
                    self.model.Add(setup_saved == 0).OnlyEnforceIf(j_follows_i.Not())
                    
                    self.setup_saved_vars.append(setup_saved)
                    
                    # 记录
                    self.setup_optimization_pairs.append({
                        'machine': machine_id,
                        'task_i': (job_i, op_idx),
                        'task_j': (job_j, op_idx),
                        'consecutive_var': j_follows_i,
                        'setup_time_saved': task_j['setup_time'],
                        'order_num': order_num,
                        'part_num': part_num,
                        'operation': op_idx + 1
                    })
                    
                    setup_count += 1
        
        print(f"  共添加 {setup_count} 个调机优化约束")
        print(f"  目标函数将激励求解器最大化调机节省")
    
    def solve(self, time_limit_seconds=300):
        """求解模型"""
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.log_search_progress = False
        self.solver.parameters.num_search_workers = 8
        
        print("\n" + "="*80)
        print("开始求解FJSP问题...")
        print(f"问题规模: {self.num_jobs}个工件, {self.num_operations}个工序, "
              f"{self.num_machines}台机器")
        print("="*80)
        
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL:
            print(f"\n✓ 找到最优解！")
        elif status == cp_model.FEASIBLE:
            print(f"\n✓ 找到可行解(时间限制内未证明最优)")
        else:
            print(f"\n❌ 未找到可行解！状态: {self.solver.StatusName()}")
            return None
        
        optimal_makespan = self.solver.Value(self.makespan)
        print(f"\n✓ 求解结果:")
        print(f"  - 最优Makespan: {optimal_makespan}")
        print(f"  - 求解时间: {self.solver.WallTime():.2f}秒")
        print(f"  - 分支数: {self.solver.NumBranches()}")
        
        # 统计调机优化效果
        setup_saved_count = 0
        total_setup_time_saved = 0
        
        print(f"\n✓ 调机优化详情:")
        for pair in self.setup_optimization_pairs:
            if self.solver.Value(pair['consecutive_var']):
                setup_saved_count += 1
                total_setup_time_saved += pair['setup_time_saved']
                
                # 打印每个优化的详细信息
                job_i, op_i = pair['task_i']
                job_j, op_j = pair['task_j']
                print(f"    ✓ M{pair['machine']}: J{job_i}→J{job_j} "
                      f"(订单{pair['order_num']}-{pair['part_num']}-工序{pair['operation']}) "
                      f"省去{pair['setup_time_saved']}分钟")
        
        if setup_saved_count > 0:
            print(f"\n✓ 调机优化效果汇总:")
            print(f"  - 省去调机次数: {setup_saved_count}")
            print(f"  - 节省调机时间: {total_setup_time_saved} 分钟")
        else:
            print(f"\n⚠ 未实现调机优化")
        
        return self.extract_solution()
    
    def extract_solution(self):
        """提取调度方案"""
        schedule = []
        
        for job_id in range(self.num_jobs):
            for op_idx in range(self.jobs[job_id]):
                start_time = self.solver.Value(self.start_vars[(job_id, op_idx)])
                end_time = self.solver.Value(self.end_vars[(job_id, op_idx)])
                
                selected_machine = None
                actual_duration = end_time - start_time
                
                for machine_id in range(self.num_machines):
                    if (job_id, op_idx, machine_id) in self.presence_vars:
                        if self.solver.Value(self.presence_vars[(job_id, op_idx, machine_id)]):
                            selected_machine = machine_id
                            
                            # 打印duration调试信息
                            if (job_id, op_idx, machine_id) in self.duration_vars:
                                duration_var, time_with, time_without = self.duration_vars[(job_id, op_idx, machine_id)]
                                solved_duration = self.solver.Value(duration_var)
                                if solved_duration < time_with:
                                    print(f"  J{job_id}-O{op_idx}-M{machine_id}: duration={solved_duration} "
                                          f"(优化! 原本={time_with})")
                            break
                
                schedule.append({
                    'job': job_id,
                    'operation': op_idx,
                    'machine': selected_machine,
                    'start': start_time,
                    'end': end_time,
                    'duration': actual_duration
                })
        
        return schedule
    
    def export_solution_to_excel(self, schedule, filename='排产结果详细分析.xlsx'):
        """导出详细的求解结果到Excel"""
        
        print(f"\n正在导出排产结果到Excel...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 工作表1: 调度方案总览
            schedule_data = []
            for task in sorted(schedule, key=lambda x: (x['start'], x['machine'], x['job'])):
                meta = self.processor.job_metadata[task['job']] if self.processor else None
                
                row = {
                    '工件ID': task['job'],
                    '工序号': task['operation'] + 1,
                    '机器编号': task['machine'],
                    '开始时间': task['start'],
                    '结束时间': task['end'],
                    '加工时长': task['duration'],
                }
                
                if meta:
                    row['订单号'] = meta['order_num']
                    row['工件号'] = meta['part_num']
                    row['批次'] = f"{meta['batch_idx']+1}/{meta['total_batches']}"
                    row['批次数量'] = meta['batch_size']
                
                schedule_data.append(row)
            
            df_schedule = pd.DataFrame(schedule_data)
            df_schedule.to_excel(writer, sheet_name='调度方案总览', index=False)
            
            # 工作表2: 批处理分析
            batch_data = []
            HEAT_TREATMENT_MACHINES = [15, 16]
            
            for pair in self.batch_processing_pairs:
                if self.solver.Value(pair['overlap_var']):
                    job_i, op_i = pair['task_i']
                    job_j, op_j = pair['task_j']
                    
                    task_i = next(t for t in schedule if t['job'] == job_i and t['operation'] == op_i)
                    task_j = next(t for t in schedule if t['job'] == job_j and t['operation'] == op_j)
                    
                    meta_i = self.processor.job_metadata[job_i]
                    meta_j = self.processor.job_metadata[job_j]
                    
                    batch_data.append({
                        '机器编号': pair['machine'],
                        '机器类型': '🔥热处理机',
                        '批次1-工件ID': job_i,
                        '批次1-订单': f"{meta_i['order_num']}-{meta_i['part_num']}",
                        '批次1-工序': op_i + 1,
                        '批次1-数量': meta_i['batch_size'],
                        '批次2-工件ID': job_j,
                        '批次2-订单': f"{meta_j['order_num']}-{meta_j['part_num']}",
                        '批次2-工序': op_j + 1,
                        '批次2-数量': meta_j['batch_size'],
                        '开始时间': task_i['start'],
                        '结束时间': task_i['end'],
                        '批处理时长': task_i['duration'],
                        '总处理数量': meta_i['batch_size'] + meta_j['batch_size'],
                        '说明': '✓ 两个批次同时在热处理机器中处理,共享设备时间'
                    })
            
            if batch_data:
                df_batch = pd.DataFrame(batch_data)
                df_batch.to_excel(writer, sheet_name='批处理详情', index=False)
            else:
                # 创建空表但有说明
                df_batch = pd.DataFrame([{'说明': '本次排产中未发现热处理批处理机会'}])
                df_batch.to_excel(writer, sheet_name='批处理详情', index=False)
            
            # 工作表3: 调机优化分析
            setup_data = []
            
            for pair in self.setup_optimization_pairs:
                if self.solver.Value(pair['consecutive_var']):
                    job_i, op_i = pair['task_i']
                    job_j, op_j = pair['task_j']
                    
                    task_i = next(t for t in schedule if t['job'] == job_i and t['operation'] == op_i)
                    task_j = next(t for t in schedule if t['job'] == job_j and t['operation'] == op_j)
                    
                    meta_i = self.processor.job_metadata[job_i]
                    meta_j = self.processor.job_metadata[job_j]
                    
                    # 检查是否真的连续
                    if task_i['end'] == task_j['start'] or task_j['end'] == task_i['start']:
                        if task_i['end'] == task_j['start']:
                            first, second = (job_i, meta_i, task_i), (job_j, meta_j, task_j)
                        else:
                            first, second = (job_j, meta_j, task_j), (job_i, meta_i, task_i)
                        
                        setup_data.append({
                            '机器编号': pair['machine'],
                            '订单-工件': f"{pair['order_num']}-{pair['part_num']}",
                            '工序号': pair['operation'],
                            '第1批-工件ID': first[0],
                            '第1批-批次': f"{first[1]['batch_idx']+1}/{first[1]['total_batches']}",
                            '第1批-数量': first[1]['batch_size'],
                            '第1批-开始': first[2]['start'],
                            '第1批-结束': first[2]['end'],
                            '第2批-工件ID': second[0],
                            '第2批-批次': f"{second[1]['batch_idx']+1}/{second[1]['total_batches']}",
                            '第2批-数量': second[1]['batch_size'],
                            '第2批-开始': second[2]['start'],
                            '第2批-结束': second[2]['end'],
                            '节省调机时间': pair['setup_time_saved'],
                            '说明': f"✓ 相同工件的连续批次,第2批省去{pair['setup_time_saved']}分钟调机时间"
                        })
            
            if setup_data:
                df_setup = pd.DataFrame(setup_data)
                df_setup.to_excel(writer, sheet_name='调机优化详情', index=False)
            else:
                df_setup = pd.DataFrame([{'说明': '本次排产中未发现调机优化机会'}])
                df_setup.to_excel(writer, sheet_name='调机优化详情', index=False)
            
            # 工作表4: 机器利用率统计
            machine_stats = {}
            makespan = self.solver.Value(self.makespan)
            
            for task in schedule:
                machine = task['machine']
                if machine not in machine_stats:
                    machine_stats[machine] = {
                        '机器编号': machine,
                        '任务数': 0,
                        '总加工时间': 0,
                        '空闲时间': makespan,
                        '利用率': 0
                    }
                
                machine_stats[machine]['任务数'] += 1
                machine_stats[machine]['总加工时间'] += task['duration']
            
            for machine in machine_stats:
                busy_time = machine_stats[machine]['总加工时间']
                machine_stats[machine]['空闲时间'] = makespan - busy_time
                machine_stats[machine]['利用率'] = f"{(busy_time / makespan * 100):.2f}%"
            
            df_machine = pd.DataFrame(list(machine_stats.values()))
            df_machine = df_machine.sort_values('机器编号')
            df_machine.to_excel(writer, sheet_name='机器利用率', index=False)
            
            # 工作表5: 优化效果汇总
            total_setup_saved = sum(
                pair['setup_time_saved'] 
                for pair in self.setup_optimization_pairs 
                if self.solver.Value(pair['consecutive_var'])
            )
            
            batch_count = sum(
                1 for pair in self.batch_processing_pairs 
                if self.solver.Value(pair['overlap_var'])
            )
            
            summary_data = [
                {'指标': '最优Makespan', '数值': makespan, '单位': '分钟'},
                {'指标': '总工件数', '数值': self.num_jobs, '单位': '个'},
                {'指标': '总工序数', '数值': self.num_operations, '单位': '道'},
                {'指标': '机器数量', '数值': self.num_machines, '单位': '台'},
                {'指标': '批处理次数', '数值': batch_count, '单位': '次'},
                {'指标': '调机优化次数', '数值': len([p for p in self.setup_optimization_pairs if self.solver.Value(p['consecutive_var'])]), '单位': '次'},
                {'指标': '节省调机时间', '数值': total_setup_saved, '单位': '分钟'},
                {'指标': '求解时间', '数值': f"{self.solver.WallTime():.2f}", '单位': '秒'},
            ]
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='优化效果汇总', index=False)
        
        print(f"✓ 排产结果已导出到: {filename}")
        print(f"  包含工作表:")
        print(f"    1. 调度方案总览 - 完整的任务时间表")
        print(f"    2. 批处理详情 - 热处理批处理的具体信息")
        print(f"    3. 调机优化详情 - 连续批次减少调机的详细记录")
        print(f"    4. 机器利用率 - 各机器的使用统计")
        print(f"    5. 优化效果汇总 - 整体优化效果")
    
    def plot_gantt_chart(self, schedule, filename='ortools_optimal_schedule.png'):
        """绘制甘特图"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(20, 12))
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_jobs))
        
        machines_schedule = {m: [] for m in range(self.num_machines)}
        for task in schedule:
            machines_schedule[task['machine']].append(task)
        
        HEAT_TREATMENT_MACHINES = [15, 16]
        
        for machine_id in range(self.num_machines):
            tasks = sorted(machines_schedule[machine_id], key=lambda x: (x['start'], x['job']))
            
            if machine_id in HEAT_TREATMENT_MACHINES and tasks:
                # 热处理机器批处理显示
                batch_groups = []
                processed = set()
                
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
                
                for batch_idx, batch in enumerate(batch_groups):
                    if len(batch) > 1:
                        total_height = 0.8
                        sub_height = total_height / len(batch)
                        
                        for sub_idx, task in enumerate(batch):
                            y_pos = machine_id - 0.4 + sub_idx * sub_height
                            color = colors[task['job']]
                            
                            rect = mpatches.Rectangle(
                                (task['start'], y_pos),
                                task['duration'],
                                sub_height * 0.95,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1.2,
                                alpha=0.85
                            )
                            ax.add_patch(rect)
                            
                            text = f"J{task['job']}-O{task['operation']}"
                            ax.text(
                                task['start'] + task['duration'] / 2,
                                y_pos + sub_height * 0.475,
                                text,
                                ha='center',
                                va='center',
                                fontsize=max(7, 10 - len(batch) // 2),
                                fontweight='bold',
                                color='white'
                            )
                        
                        batch_rect = mpatches.Rectangle(
                            (batch[0]['start'], machine_id - 0.4),
                            batch[0]['duration'],
                            total_height,
                            facecolor='none',
                            edgecolor='red',
                            linewidth=3,
                            linestyle='--',
                            alpha=0.9
                        )
                        ax.add_patch(batch_rect)
                    else:
                        task = batch[0]
                        y_pos = machine_id - 0.4
                        color = colors[task['job']]
                        
                        rect = mpatches.Rectangle(
                            (task['start'], y_pos),
                            task['duration'],
                            0.8,
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=0.85
                        )
                        ax.add_patch(rect)
                        
                        text = f"J{task['job']}-O{task['operation']}"
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
            else:
                for task in tasks:
                    color = colors[task['job']]
                    
                    rect = mpatches.Rectangle(
                        (task['start'], machine_id - 0.4),
                        task['duration'],
                        0.8,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.85
                    )
                    ax.add_patch(rect)
                    
                    text = f"J{task['job']}-O{task['operation']}"
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
        
        makespan = self.solver.Value(self.makespan)
        ax.set_xlabel('时间', fontsize=14, fontweight='bold')
        ax.set_ylabel('机器', fontsize=14, fontweight='bold')
        ax.set_title(
            f'OR-Tools求解的FJSP最优调度方案 (Makespan={makespan})\n'
            f'🔥 红色虚线框 = 热处理批处理', 
            fontsize=16, fontweight='bold', pad=20
        )
        
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([
            f'M{i} 🔥(热处理)' if i in HEAT_TREATMENT_MACHINES else f'M{i}' 
            for i in range(self.num_machines)
        ], fontsize=11)
        ax.set_ylim(-0.5, self.num_machines - 0.5)
        ax.set_xlim(-15, makespan * 1.05)
        
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f'工件 {i}', alpha=0.85) 
            for i in range(self.num_jobs)
        ]
        legend_patches.append(
            mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2.5, 
                          linestyle='--', label='🔥 批处理组')
        )
        
        ax.legend(
            handles=legend_patches, 
            loc='upper right', 
            fontsize=10, 
            ncol=2 if self.num_jobs > 6 else 1,
            framealpha=0.95,
            edgecolor='black'
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ 甘特图已保存: {filename}")
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