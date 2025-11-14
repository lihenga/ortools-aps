"""
测试热处理类型读取是否正确
"""

from Pre import OrderProcessor

def test_heat_type():
    processor = OrderProcessor(excel_folder='dingdan')
    processor.process_all_orders(max_batch_size=40)
    
    print("\n" + "="*80)
    print("热处理类型验证")
    print("="*80)
    
    for job_id, meta in processor.job_metadata.items():
        print(f"\n工件 J{job_id} ({meta['order_num']}-{meta['part_num']}, 批次{meta['batch_idx']}):")
        
        for op_idx, operation in enumerate(meta['operations']):
            machines = operation['available_machines']
            
            # 检查是否是热处理工序
            is_heat = any(m in [10, 11, 12] for m in machines)
            
            if is_heat:
                heat_type = operation.get('heat_treatment_type')
                print(f"  工序{op_idx}: 机器{machines}, 热处理类型={heat_type}")

if __name__ == '__main__':
    test_heat_type()