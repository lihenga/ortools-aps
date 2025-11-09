"""
调试工具:导出Excel读取的详细信息
"""

from Pre import OrderProcessor
import pandas as pd

def debug_excel(filepath='dingdan/10401.xlsx'):
    """详细检查Excel文件内容"""
    
    print("="*80)
    print(f"调试Excel文件: {filepath}")
    print("="*80)
    
    # 读取原始数据
    df = pd.read_excel(filepath, header=None)
    
    print(f"\nExcel原始数据 (共{len(df)}行 × {len(df.columns)}列):")
    print("-"*80)
    
    for idx in range(min(25, len(df))):  # 显示前25行
        row = df.iloc[idx]
        print(f"第{idx+1:2d}行: ", end="")
        
        if idx == 0:
            print(f"订单号 = {row[0]}")
        elif idx == 1:
            print(f"工件号 = {row[0]}")
        elif idx == 2:
            print(f"数量 = {row[0]}")
        else:
            # 工序行
            op_num = idx - 2
            print(f"工序{op_num:2d}: ", end="")
            
            # 检查每一列
            for col_idx in range(min(4, len(row))):
                val = row[col_idx]
                if pd.isna(val):
                    print(f"列{col_idx+1}=空 ", end="")
                else:
                    print(f"列{col_idx+1}={val} ", end="")
            print()
    
    print("\n" + "="*80)
    print("使用OrderProcessor读取:")
    print("="*80)
    
    processor = OrderProcessor(excel_folder=filepath)
    order = processor.read_excel_file(filepath)
    
    if order:
        print(f"\n✓ 成功读取订单")
        print(f"  订单号: {order['order_num']}")
        print(f"  工件号: {order['part_num']}")
        print(f"  数量: {order['quantity']}")
        print(f"  工序数: {len(order['operations'])}")
        
        print("\n工序详情:")
        print("-"*80)
        for i, op in enumerate(order['operations']):
            print(f"  工序{i+1:2d}: 调机={op['setup_time']:6.1f}, "
                  f"固定={op['fixed_time']:6.1f}, "
                  f"单件={op['unit_time']:6.1f}, "
                  f"机器={op['available_machines']}")
    else:
        print("\n❌ 读取失败")
    
    print("="*80)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        debug_excel(sys.argv[1])
    else:
        # 默认检查dingdan文件夹中的第一个文件
        import os
        if os.path.exists('dingdan'):
            files = [f for f in os.listdir('dingdan') if f.endswith(('.xlsx', '.xls'))]
            if files:
                debug_excel(f'dingdan/{files[0]}')
            else:
                print("dingdan文件夹中没有Excel文件")
        else:
            print("dingdan文件夹不存在")