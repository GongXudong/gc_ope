from gc_ope.env.utils.my_maze.get_grid_inner_points import generate_grid_points


# 测试函数
def test_grid_points():
    """测试生成网格点的函数"""
    # 你的测试用例：中心点(0,0)，边长4，4x4网格
    center_x, center_y = 0, 0
    maze_size_scaling = 4
    n = 4
    
    points = generate_grid_points(center_x, center_y, maze_size_scaling, n)
    print(f"中心点: ({center_x}, {center_y})")
    print(f"正方形边长: {maze_size_scaling}")
    print(f"网格大小: {n}x{n}")
    print("生成的坐标点:")
    
    # 按行打印点（从顶部开始，对应你的描述顺序）
    for i in range(n-1, -1, -1):  # 从最后一行开始（顶部）
        row_points = points[i*n:(i+1)*n]
        row_label = "上" if i == n-1 else "下" if i == 0 else f"第{i+1}"
        print(f"{row_label}行: {row_points}")
    
    # 验证关键点
    print("\n关键点验证:")
    print(f"最左上角点: {points[3*n]}")      # 第一行的第一个点
    print(f"最右上角点: {points[4*n-1]}")    # 第一行的最后一个点
    print(f"最左下角点: {points[0]}")        # 最后一行的第一个点
    print(f"最右下角点: {points[n-1]}")      # 最后一行的最后一个点
    
    print("\n" + "="*50 + "\n")
    
    # 测试其他情况
    test_cases = [
        (0, 0, 4, 2),
        (0, 0, 4, 3),
        (2, 3, 6, 3)
    ]
    
    for cx, cy, size, grid_n in test_cases:
        points = generate_grid_points(cx, cy, size, grid_n)
        print(f"中心({cx},{cy}), 边长{size}, {grid_n}x{grid_n}网格:")
        print(f"第一个点: {points[0]}, 最后一个点: {points[-1]}")
        print()

# 可视化函数
def visualize_grid_points(center_x, center_y, maze_size_scaling, n):
    """
    可视化生成的网格点
    """
    try:
        import matplotlib.pyplot as plt

        points = generate_grid_points(center_x, center_y, maze_size_scaling, n)

        # 提取x和y坐标
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # 绘制正方形边界
        half_size = maze_size_scaling / 2
        square_x = [center_x - half_size, center_x + half_size, 
                   center_x + half_size, center_x - half_size, center_x - half_size]
        square_y = [center_y - half_size, center_y - half_size, 
                   center_y + half_size, center_y + half_size, center_y - half_size]

        plt.figure(figsize=(10, 10))
        plt.plot(square_x, square_y, 'b-', linewidth=2, label='正方形边界')
        plt.scatter(x_coords, y_coords, c='red', s=50, label='网格点')
        plt.scatter([center_x], [center_y], c='green', s=100, marker='*', label='中心点')

        # 标注关键点
        plt.annotate(f'左上({points[(n-1)*n][0]:.1f},{points[(n-1)*n][1]:.1f})', 
                    xy=points[(n-1)*n], xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'右上({points[n*n-1][0]:.1f},{points[n*n-1][1]:.1f})', 
                    xy=points[n*n-1], xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'左下({points[0][0]:.1f},{points[0][1]:.1f})', 
                    xy=points[0], xytext=(10, -20), textcoords='offset points')
        plt.annotate(f'右下({points[n-1][0]:.1f},{points[n-1][1]:.1f})', 
                    xy=points[n-1], xytext=(10, -20), textcoords='offset points')
    
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title(f'正方形内部{n}x{n}网格点分布 (间隔: {maze_size_scaling/n:.2f})')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    except ImportError:
        print("要可视化结果，请安装matplotlib: pip install matplotlib")

if __name__ == "__main__":
    # 运行测试
    test_grid_points()

    # 可视化你的例子
    print("可视化你的例子:")
    visualize_grid_points(0, 0, 8, 6)
