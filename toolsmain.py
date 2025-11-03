"""
简单直接运行OR-Tools求解FJSP并显示结果
"""

from tools import FJSPSolver

def main():
    
    # 创建求解器
    solver = FJSPSolver()
    
    # 构建模型
    print("\n正在构建优化模型...")
    solver.build_model()
    
    # 求解（限时5分钟）
    print("开始求解...(限时5分钟)\n")
    schedule = solver.solve(time_limit_seconds=300)
    
    if schedule:
        # 打印详细方案
        solver.print_schedule(schedule)
        
        # 生成甘特图
        solver.plot_gantt_chart(schedule, 'ortools_optimal_schedule.png')
        
        # 最终结果
        makespan = solver.solver.Value(solver.makespan)
        status = solver.solver.StatusName()
        
        print("\n\n" + "="*30)
        print(f"求解完成！")
        print(f"最优Makespan: {makespan}")
        print(f"求解状态: {status}")
        print(f"甘特图已保存: ortools_optimal_schedule.png")
        print("="*30)
    else:
        print("\n 求解失败，未找到可行解")

if __name__ == '__main__':
    main()