CREATE DATABASE IF NOT EXISTS supermarket;
USE supermarket;

CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    barcode VARCHAR(20) UNIQUE,
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO products (barcode, product_name, price, category) VALUES
('690123456789', '蒙牛纯牛奶', 5.50, 'dairy'),
('692345678901', '乐事薯片', 8.90, 'snacks'),
('691234567890', '金龙鱼食用油', 65.00, 'oil'),
('693456789012', '农夫山泉矿泉水', 2.00, 'water'),
('694567890123', '桃李面包', 7.50, 'bread');