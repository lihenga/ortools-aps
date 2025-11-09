"""
简单直接运行OR-Tools求解FJSP并显示结果
从Excel文件读取订单信息进行排产
"""

from tools import FJSPSolver
from Pre import OrderProcessor
from datetime import datetime

def main():
    # 步骤1: 从Excel读取订单数据
    print("="*60)
    print("步骤1: 读取Excel订单数据")
    print("="*60)
    
    # 可以指定文件夹或单个文件
    processor = OrderProcessor(excel_folder='dingdan')
    processor.process_all_orders(max_batch_size=40)
    processor.print_summary()
    
    # 导出输入数据验证
    processor.export_to_excel('0_输入数据验证.xlsx')
    
    # 步骤2: 创建求解器并传入processor
    print("\n" + "="*60)
    print("步骤2: 构建优化模型")
    print("="*60)
    
    solver = FJSPSolver(processor=processor)
    solver.build_model()
    
    # 步骤3: 求解
    print("\n" + "="*60)
    print("步骤3: 开始求解 (限时300秒)")
    print("="*60)
    
    schedule = solver.solve(time_limit_seconds=60)
    
    if schedule:
        # 打印详细方案
        solver.print_schedule(schedule)
        
        # 导出详细Excel结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f'1_排产结果详细分析_{timestamp}.xlsx'
        solver.export_solution_to_excel(schedule, excel_filename)
        
        # 生成甘特图
        gantt_filename = f'2_甘特图_{timestamp}.png'
        solver.plot_gantt_chart(schedule, gantt_filename)
        
        # 最终结果
        makespan = solver.solver.Value(solver.makespan)
        status = solver.solver.StatusName()
        
        print("\n\n" + "="*60)
        print(f"✓ 求解完成！")
        print(f"  最优Makespan: {makespan}")
        print(f"  求解状态: {status}")
        print(f"  Excel结果: {excel_filename}")
        print(f"  甘特图: {gantt_filename}")
        print("="*60)
        
        print("\n📊 Excel文件包含以下工作表:")
        print("  1️⃣  调度方案总览 - 每个任务的详细时间安排")
        print("  2️⃣  批处理详情 - 热处理机器(15,16)的批处理情况")
        print("  3️⃣  调机优化详情 - 同一工件连续批次的调机时间节省")
        print("  4️⃣  机器利用率 - 各机器的工作负荷统计")
        print("  5️⃣  优化效果汇总 - 整体优化指标")
        
    else:
        print("\n❌ 求解失败，未找到可行解")

if __name__ == '__main__':
    main()