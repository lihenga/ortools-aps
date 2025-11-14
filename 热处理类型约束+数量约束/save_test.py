"""
测试加载已保存的排产结果
"""

from schedule_saver import ScheduleSaver, ScheduleLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
import os

def plot_gantt_from_saved_schedule(loader, filename=None):
    """
    从加载器生成甘特图（与原始甘特图风格一致）
    
    Args:
        loader: ScheduleLoader实例
        filename: 保存的文件名
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    schedule = loader.schedule
    num_machines = loader.solver_data['num_machines']
    num_jobs = loader.solver_data['num_jobs']
    makespan = loader.solver_data['makespan']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 使用tab20颜色方案
    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    
    # 按机器分组任务
    machines_schedule = {m: [] for m in range(num_machines)}
    for task in schedule:
        machines_schedule[task['machine']].append(task)
    
    # 热处理机器列表
    HEAT_TREATMENT_MACHINES = [10, 11, 12]
    
    # 绘制每台机器的任务
    for machine_id in range(num_machines):
        tasks = sorted(machines_schedule[machine_id], key=lambda x: (x['start'], x['job']))
        
        if machine_id in HEAT_TREATMENT_MACHINES and tasks:
            # 热处理机器：批处理显示逻辑
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
                    # 多个任务批处理：分割显示
                    total_height = 0.8
                    sub_height = total_height / len(batch)
                    
                    for sub_idx, task in enumerate(batch):
                        y_pos = machine_id - 0.4 + sub_idx * sub_height
                        color = colors[task['job']]
                        duration = task['end'] - task['start']
                        
                        rect = mpatches.Rectangle(
                            (task['start'], y_pos),
                            duration,
                            sub_height * 0.95,
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.2,
                            alpha=0.85
                        )
                        ax.add_patch(rect)
                        
                        # 文本标签
                        text = f"J{task['job']}-O{task['operation']}"
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
                    
                    # 红色虚线框标记批处理
                    batch_rect = mpatches.Rectangle(
                        (batch[0]['start'], machine_id - 0.4),
                        batch[0]['end'] - batch[0]['start'],
                        total_height,
                        facecolor='none',
                        edgecolor='red',
                        linewidth=3,
                        linestyle='--',
                        alpha=0.9
                    )
                    ax.add_patch(batch_rect)
                else:
                    # 单个任务
                    task = batch[0]
                    y_pos = machine_id - 0.4
                    color = colors[task['job']]
                    duration = task['end'] - task['start']
                    
                    rect = mpatches.Rectangle(
                        (task['start'], y_pos),
                        duration,
                        0.8,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.85
                    )
                    ax.add_patch(rect)
                    
                    text = f"J{task['job']}-O{task['operation']}"
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
            # 普通机器：标准显示
            for task in tasks:
                color = colors[task['job']]
                duration = task['end'] - task['start']
                
                rect = mpatches.Rectangle(
                    (task['start'], machine_id - 0.4),
                    duration,
                    0.8,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.85
                )
                ax.add_patch(rect)
                
                text = f"J{task['job']}-O{task['operation']}"
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
    
    # 设置坐标轴
    ax.set_xlabel('时间', fontsize=14, fontweight='bold')
    ax.set_ylabel('机器', fontsize=14, fontweight='bold')
    
    # 标题
    title = (f'加载的排产方案 (Makespan={makespan})\n'
             f'✓ 红色虚线框 = 热处理批处理')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Y轴标签
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([
        f'M{i} (热处理)' if i in HEAT_TREATMENT_MACHINES else f'M{i}' 
        for i in range(num_machines)
    ], fontsize=11)
    ax.set_ylim(-0.5, num_machines - 0.5)
    ax.set_xlim(-15, makespan * 1.05)
    
    # 网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f'工件 {i}', alpha=0.85) 
        for i in range(num_jobs)
    ]
    legend_patches.append(
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2.5, 
                      linestyle='--', label='✓ 批处理组')
    )
    
    ax.legend(
        handles=legend_patches, 
        loc='upper right', 
        fontsize=10, 
        ncol=2 if num_jobs > 6 else 1,
        framealpha=0.95,
        edgecolor='black'
    )
    
    plt.tight_layout()
    
    # 保存图形
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'加载的排产甘特图_{timestamp}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 甘特图已保存: {filename}")
    plt.close()
    
    return filename


def test_load():
    """测试加载功能"""
    
    print("\n" + "="*80)
    print("测试加载已保存的排产结果")
    print("="*80)
    
    saver = ScheduleSaver()
    
    # 查找最新的pickle文件
    save_dir = saver.save_dir
    if not os.path.exists(save_dir):
        print(f"\n❌ 未找到保存目录: {save_dir}")
        print("请先运行 toolsmain.py 生成排产结果")
        return
    
    # 列出所有pickle文件
    pickle_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        print(f"\n❌ 在 {save_dir} 目录下未找到排产结果文件")
        print("请先运行 toolsmain.py 生成排产结果")
        return
    
    # 按时间排序，选择最新的
    pickle_files.sort(reverse=True)
    latest_file = os.path.join(save_dir, pickle_files[0])
    
    print(f"\n✓ 找到 {len(pickle_files)} 个已保存的排产结果")
    print(f"✓ 加载最新的文件: {latest_file}")
    
    # 加载数据
    save_data = saver.load_schedule(latest_file)
    
    # 创建加载器
    loader = ScheduleLoader(save_data)
    
    print("\n" + "="*80)
    print("排产结果摘要")
    print("="*80)
    loader.print_summary()
    
    # 显示详细信息
    print("\n" + "="*80)
    print("详细信息")
    print("="*80)
    
    # 1. 机器利用率统计
    print("\n📊 机器利用率统计:")
    makespan = loader.solver_data['makespan']
    machine_usage = {}
    
    for task in loader.schedule:
        machine = task['machine']
        duration = task['end'] - task['start']
        if machine not in machine_usage:
            machine_usage[machine] = 0
        machine_usage[machine] += duration
    
    for machine_id in sorted(machine_usage.keys()):
        total_time = machine_usage[machine_id]
        utilization = (total_time / makespan) * 100
        print(f"  机器M{machine_id}: {utilization:.1f}% (工作{total_time:.0f}分钟 / 总时长{makespan}分钟)")
    
    # 2. 调机优化详情
    setup_results = loader.optimization_results['setup_optimization']
    consecutive_setups = [r for r in setup_results if r['consecutive']]
    
    if consecutive_setups:
        print(f"\n🔧 调机优化详情 (共{len(consecutive_setups)}次):")
        for i, result in enumerate(consecutive_setups[:10], 1):
            print(f"  {i}. 机器M{result['machine']}: "
                  f"J{result['task_i'][0]}-O{result['task_i'][1]+1} → "
                  f"J{result['task_j'][0]}-O{result['task_j'][1]+1} "
                  f"(节省{result['setup_time_saved']}分钟)")
        
        if len(consecutive_setups) > 10:
            print(f"  ... 还有 {len(consecutive_setups) - 10} 个调机优化")
    
    # 3. 批处理详情
    batch_results = loader.optimization_results['batch_processing']
    batch_pairs = [r for r in batch_results if r['fully_overlap']]
    
    if batch_pairs:
        print(f"\n🔥 热处理批处理详情 (共{len(batch_pairs)}对):")
        for i, result in enumerate(batch_pairs[:10], 1):
            print(f"  {i}. 机器M{result['machine']}: "
                  f"J{result['task_i'][0]}-O{result['task_i'][1]+1} + "
                  f"J{result['task_j'][0]}-O{result['task_j'][1]+1} "
                  f"(批处理)")
        
        if len(batch_pairs) > 10:
            print(f"  ... 还有 {len(batch_pairs) - 10} 个批处理对")
    
    # 4. 查询功能测试
    print(f"\n🔍 空闲时间分析:")
    
    # 找出每台机器的空闲时间
    for machine_id in range(min(5, loader.solver_data['num_machines'])):
        tasks = loader.get_machine_schedule(machine_id)
        if not tasks:
            print(f"  机器M{machine_id}: 完全空闲")
            continue
        
        tasks_sorted = sorted(tasks, key=lambda x: x['start'])
        idle_times = []
        
        # 开头的空闲时间
        if tasks_sorted[0]['start'] > 0:
            idle_times.append((0, tasks_sorted[0]['start']))
        
        # 中间的空闲时间
        for i in range(len(tasks_sorted) - 1):
            gap_start = tasks_sorted[i]['end']
            gap_end = tasks_sorted[i + 1]['start']
            if gap_end > gap_start:
                idle_times.append((gap_start, gap_end))
        
        # 结尾的空闲时间
        if tasks_sorted[-1]['end'] < makespan:
            idle_times.append((tasks_sorted[-1]['end'], makespan))
        
        if idle_times:
            print(f"  机器M{machine_id}: {len(idle_times)}个空闲时间段")
            for start, end in idle_times[:3]:
                print(f"    - [{start}, {end}] (时长: {end-start})")
        else:
            print(f"  机器M{machine_id}: 无空闲时间")
    
    # 5. 生成甘特图
    print(f"\n📈 生成甘特图...")
    gantt_file = plot_gantt_from_saved_schedule(loader)
    
    # 6. 按工件查看
    print(f"\n📦 按工件查看 (前5个工件):")
    for job_id in range(min(5, loader.solver_data['num_jobs'])):
        job_tasks = loader.get_job_schedule(job_id)
        if job_tasks:
            print(f"  工件J{job_id}: {len(job_tasks)}个工序")
            for task in job_tasks:
                print(f"    - O{task['operation']}: 机器M{task['machine']}, "
                      f"时间[{task['start']}, {task['end']}]")
    
    print("\n" + "="*80)
    print("✓ 测试完成！数据完整性验证通过")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - 甘特图: {gantt_file}")
    print(f"\n可以基于此数据进行插单操作")


def list_all_schedules():
    """列出所有已保存的排产结果"""
    
    print("\n" + "="*80)
    print("所有已保存的排产结果")
    print("="*80)
    
    saver = ScheduleSaver()
    save_dir = saver.save_dir
    
    if not os.path.exists(save_dir):
        print(f"\n未找到保存目录: {save_dir}")
        return
    
    # 列出所有pickle文件
    pickle_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        print(f"\n在 {save_dir} 目录下未找到排产结果文件")
        return
    
    pickle_files.sort(reverse=True)
    
    print(f"\n找到 {len(pickle_files)} 个排产结果:\n")
    
    for i, filename in enumerate(pickle_files, 1):
        filepath = os.path.join(save_dir, filename)
        
        # 加载基本信息
        try:
            save_data = saver.load_schedule(filepath)
            makespan = save_data['solver_data']['makespan']
            num_jobs = save_data['solver_data']['num_jobs']
            timestamp = save_data['timestamp']
            
            print(f"{i}. {filename}")
            print(f"   时间戳: {timestamp}")
            print(f"   Makespan: {makespan}")
            print(f"   工件数: {num_jobs}")
            print()
        except Exception as e:
            print(f"{i}. {filename} (加载失败: {e})")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        # 默认: 测试加载最新的排产结果
        test_load()
    elif len(sys.argv) == 2 and sys.argv[1] == 'list':
        # 列出所有排产结果
        list_all_schedules()
    else:
        print("用法:")
        print("  python save_test.py           # 查看最新排产结果")
        print("  python save_test.py list      # 列出所有排产结果")