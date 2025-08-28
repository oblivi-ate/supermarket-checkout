from kafka import KafkaConsumer
import json

def process_payment(order):
    print(f"\n处理支付订单: {order['order_id']}")
    print("商品清单:")
    for product, info in order['items'].items():
        print(f"- {product}: {info['quantity']} x ¥{info['price']:.2f}")
    print(f"总计: ¥{order['total']:.2f}")
    # 模拟调用支付网关
    print("调用支付网关接口...")
    print("支付成功!\n")

if __name__ == "__main__":
    consumer = KafkaConsumer(
        'checkout_orders',
        bootstrap_servers='127.0.0.1:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    print("支付系统已启动，等待订单...")
    for message in consumer:
        order = message.value
        process_payment(order)