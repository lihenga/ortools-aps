"""
排产结果保存工具
保存完整的排产结果用于后续插单分析
"""

import pickle
import json
from datetime import datetime
import pandas as pd

class ScheduleSaver:
    def __init__(self, save_dir='saved_schedules'):
        """
        初始化保存器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def save_complete_schedule(self, schedule, solver, base_filename='排产结果'):
        """
        保存完整的排产结果（包含用于插单的中间数据）
        
        Args:
            schedule: 调度方案列表
            solver: FJSPSolver实例
            base_filename: 基础文件名
        
        Returns:
            dict: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ✅ 提取调机优化结果（只保存值，不保存变量）
        setup_results = []
        for pair in solver.setup_optimization_pairs:
            consecutive_value = solver.solver.Value(pair['consecutive_var'])
            setup_results.append({
                'machine': pair['machine'],
                'task_i': pair['task_i'],
                'task_j': pair['task_j'],
                'consecutive': consecutive_value,
                'setup_time_saved': pair['setup_time_saved'],
                'order_num': pair.get('order_num', ''),
                'part_num': pair.get('part_num', ''),
                'operation': pair.get('operation', 0)
            })
        
        # ✅ 提取批处理结果（只保存值，不保存变量）
        batch_results = []
        for pair in solver.batch_processing_pairs:
            overlap_value = solver.solver.Value(pair['overlap_var'])
            batch_results.append({
                'machine': pair['machine'],
                'task_i': pair['task_i'],
                'task_j': pair['task_j'],
                'fully_overlap': overlap_value
            })
        
        # ✅ 提取调机节省总时间（只保存值，不保存变量）
        total_setup_saved = 0
        if solver.setup_saved_vars:
            for var in solver.setup_saved_vars:
                total_setup_saved += solver.solver.Value(var)
        
        # 1. 保存 Pickle 格式（完整的 Python 对象，用于插单）
        pickle_file = f'{self.save_dir}/{base_filename}_{timestamp}.pkl'
        save_data = {
            'schedule': schedule,
            'solver_data': {
                'num_jobs': solver.num_jobs,
                'num_machines': solver.num_machines,
                'num_operations': solver.num_operations,
                'jobs': solver.jobs,
                'makespan': solver.solver.Value(solver.makespan),
                'status': solver.solver.StatusName(),
                'horizon': solver.horizon,
            },
            'processor_data': None,
            'optimization_results': {
                'setup_optimization': setup_results,
                'batch_processing': batch_results,
                'total_setup_saved': total_setup_saved,
            },
            'timestamp': timestamp
        }
        
        # 如果有 processor，保存相关数据
        if solver.processor is not None:
            save_data['processor_data'] = {
                'job_metadata': solver.processor.job_metadata,
                'processing_time': solver.processor.processing_time,
                'processing_time_no_setup': solver.processor.processing_time_no_setup,
            }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Pickle文件已保存: {pickle_file}")
        
        # 2. 保存 JSON 格式（可读性好，用于分析）
        json_file = f'{self.save_dir}/{base_filename}_{timestamp}.json'
        json_data = {
            'schedule': schedule,
            'solver_info': {
                'num_jobs': solver.num_jobs,
                'num_machines': solver.num_machines,
                'num_operations': solver.num_operations,
                'makespan': solver.solver.Value(solver.makespan),
                'status': solver.solver.StatusName(),
            },
            'optimization_results': {
                'setup_optimization_count': len([r for r in setup_results if r['consecutive']]),
                'batch_processing_count': len([r for r in batch_results if r['fully_overlap']]),
                'total_setup_saved': total_setup_saved,
            },
            'timestamp': timestamp
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ JSON文件已保存: {json_file}")
        
        # 3. 保存 Excel 报告
        excel_file = f'{base_filename}_{timestamp}.xlsx'
        self._export_to_excel(schedule, solver, excel_file, setup_results, batch_results)
        print(f"✓ Excel报告已保存: {excel_file}")
        
        # 4. 保存甘特图
        gantt_file = f'{base_filename}_甘特图_{timestamp}.png'
        solver.plot_gantt_chart(schedule, gantt_file)
        print(f"✓ 甘特图已保存: {gantt_file}")
        
        # 5. 保存文本报告
        text_file = f'{base_filename}_报告_{timestamp}.txt'
        self._export_text_report(schedule, solver, text_file, setup_results, batch_results)
        print(f"✓ 文本报告已保存: {text_file}")
        
        return {
            'pickle': pickle_file,
            'json': json_file,
            'excel': excel_file,
            'gantt': gantt_file,
            'text': text_file
        }
    
    def load_schedule(self, pickle_file):
        """
        加载已保存的排产结果
        
        Args:
            pickle_file: pickle文件路径
        
        Returns:
            dict: 保存的数据
        """
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    def _export_to_excel(self, schedule, solver, filename, setup_results, batch_results):
        """导出详细的Excel报告"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. 调度方案总览
            df_schedule = pd.DataFrame(schedule)
            df_schedule.to_excel(writer, sheet_name='调度方案总览', index=False)
            
            # 2. 调机优化详情
            if setup_results:
                setup_data = []
                for result in setup_results:
                    if result['consecutive']:
                        setup_data.append({
                            '机器编号': result['machine'],
                            '前序工件': f"J{result['task_i'][0]}-O{result['task_i'][1]+1}",
                            '后续工件': f"J{result['task_j'][0]}-O{result['task_j'][1]+1}",
                            '订单号': result['order_num'],
                            '工件编号': result['part_num'],
                            '工序': result['operation'],
                            '节省调机时间': result['setup_time_saved'],
                            '是否连续': '是'
                        })
                
                if setup_data:
                    df_setup = pd.DataFrame(setup_data)
                    df_setup.to_excel(writer, sheet_name='调机优化详情', index=False)
            
            # 3. 批处理详情
            if batch_results:
                batch_data = []
                for result in batch_results:
                    if result['fully_overlap']:
                        batch_data.append({
                            '机器编号': result['machine'],
                            '工件1': f"J{result['task_i'][0]}-O{result['task_i'][1]+1}",
                            '工件2': f"J{result['task_j'][0]}-O{result['task_j'][1]+1}",
                            '批处理': '是'
                        })
                
                if batch_data:
                    df_batch = pd.DataFrame(batch_data)
                    df_batch.to_excel(writer, sheet_name='批处理详情', index=False)
            
            # 4. 机器利用率
            machine_usage = {}
            for task in schedule:
                machine = task['machine']
                duration = task['end'] - task['start']
                if machine not in machine_usage:
                    machine_usage[machine] = 0
                machine_usage[machine] += duration
            
            makespan = solver.solver.Value(solver.makespan)
            usage_data = []
            for machine, total_time in sorted(machine_usage.items()):
                usage_data.append({
                    '机器编号': machine,
                    '总工作时间': total_time,
                    '利用率': f"{total_time/makespan*100:.1f}%"
                })
            
            df_usage = pd.DataFrame(usage_data)
            df_usage.to_excel(writer, sheet_name='机器利用率', index=False)
            
            # 5. 优化效果汇总
            total_setup_saved = sum(r['setup_time_saved'] for r in setup_results if r['consecutive'])
            summary_data = {
                '指标': [
                    '最优Makespan',
                    '工件总数',
                    '机器总数',
                    '工序总数',
                    '调机优化次数',
                    '调机节省总时间',
                    '批处理次数'
                ],
                '数值': [
                    makespan,
                    solver.num_jobs,
                    solver.num_machines,
                    solver.num_operations,
                    len([r for r in setup_results if r['consecutive']]),
                    total_setup_saved,
                    len([r for r in batch_results if r['fully_overlap']])
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='优化效果汇总', index=False)
    
    def _export_text_report(self, schedule, solver, filename, setup_results, batch_results):
        """导出文本格式报告"""
        makespan = solver.solver.Value(solver.makespan)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("排产结果详细报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最优Makespan: {makespan}\n")
            f.write(f"工件总数: {solver.num_jobs}\n")
            f.write(f"机器总数: {solver.num_machines}\n")
            f.write(f"工序总数: {solver.num_operations}\n\n")
            
            # 调机优化统计
            setup_count = len([r for r in setup_results if r['consecutive']])
            total_setup_saved = sum(r['setup_time_saved'] for r in setup_results if r['consecutive'])
            f.write(f"调机优化次数: {setup_count}\n")
            f.write(f"调机节省总时间: {total_setup_saved}\n\n")
            
            # 批处理统计
            batch_count = len([r for r in batch_results if r['fully_overlap']])
            f.write(f"批处理次数: {batch_count}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("详细调度方案:\n")
            f.write("-"*80 + "\n\n")
            
            for task in sorted(schedule, key=lambda x: (x['job'], x['operation'])):
                f.write(f"工件J{task['job']}-工序O{task['operation']}: "
                       f"机器M{task['machine']}, "
                       f"时间[{task['start']}, {task['end']}], "
                       f"时长{task['end']-task['start']}\n")


class ScheduleLoader:
    """加载已保存的排产结果"""
    
    def __init__(self, save_data):
        """
        初始化加载器
        
        Args:
            save_data: 从pickle加载的数据
        """
        self.schedule = save_data['schedule']
        self.solver_data = save_data['solver_data']
        self.processor_data = save_data['processor_data']
        self.optimization_results = save_data['optimization_results']
        self.timestamp = save_data['timestamp']
    
    def get_machine_schedule(self, machine_id):
        """获取某台机器的所有任务"""
        return [task for task in self.schedule if task['machine'] == machine_id]
    
    def get_job_schedule(self, job_id):
        """获取某个工件的所有任务"""
        return sorted([task for task in self.schedule if task['job'] == job_id], 
                     key=lambda x: x['operation'])
    
    def get_time_window(self, start_time, end_time):
        """获取某个时间窗口内的所有任务"""
        return [task for task in self.schedule 
                if task['start'] < end_time and task['end'] > start_time]
    
    def print_summary(self):
        """打印摘要信息"""
        print(f"\n排产结果摘要:")
        print(f"  时间戳: {self.timestamp}")
        print(f"  Makespan: {self.solver_data['makespan']}")
        print(f"  工件数: {self.solver_data['num_jobs']}")
        print(f"  机器数: {self.solver_data['num_machines']}")
        print(f"  任务总数: {len(self.schedule)}")
        print(f"  调机优化次数: {len([r for r in self.optimization_results['setup_optimization'] if r['consecutive']])}")
        print(f"  批处理次数: {len([r for r in self.optimization_results['batch_processing'] if r['fully_overlap']])}")
        print(f"  调机节省总时间: {self.optimization_results['total_setup_saved']}")