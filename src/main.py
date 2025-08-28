import cv2
from .checkout_system import CheckoutSystem

def main():
    system = CheckoutSystem()
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    
    print("自助收银系统已启动")
    print("操作说明:")
    print("1. 将商品展示在摄像头前")
    print("2. 按 'c' 键结账")
    print("3. 按 ESC 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头错误")
            break
            
        processed_frame = system.process_frame(frame)
        cv2.imshow('Supermarket Self-Checkout', processed_frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):  # 结账
            receipt = system.checkout()
            print(f"订单提交成功! 订单号: {receipt['order_id']}")
            # 显示结账成功提示
            cv2.putText(processed_frame, "Payment Successful!", (150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Supermarket Self-Checkout', processed_frame)
            cv2.waitKey(2000)  # 显示2秒
        elif key == 27:  # ESC退出
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("系统已关闭")

if __name__ == "__main__":
    main()