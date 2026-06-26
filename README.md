# ImgTools
toolbox for image processing written in Python
using python3.7
using opencv4

---

## labelpose.py — YOLO Pose 标注工具

一个基于 Tkinter 的 YOLO Pose 标注工具，支持多类别、多关键点数量的目标框 + 关键点标注，标注结果直接导出为标准 YOLO Pose 格式。

### 快速开始

```bash
python labelpose.py
```

### 操作流程

1. 点击 **openDir** 打开图片所在目录，右侧列表会按数字自然排序显示所有图片
2. 点击左侧列表中的图片名加载图片，支持滚轮缩放、拖拽平移
3. 点击 **Create**（或 `Ctrl+R`），在弹出的对话框中**选择类别**
4. 按顺序标注：
   - 第 1 次左键点击：目标框左上角
   - 第 2 次左键点击：目标框右下角
   - 之后每次左键点击：依次标注关键点（`visible=2`，可见）
   - **中键**点击：遮挡但仍可定位的关键点（`visible=1`，白色编号）
   - **右键**点击：缺失/不可见的关键点（`visible=0`，不显示）
5. 标完该类别的所有关键点后自动保存，返回平移模式
6. 按 `F8` 撤销上一个目标

### 配置文件：labelpose.config

```json
{
  "class_names": ["shuttlecock", "racket"],
  "class_colors": ["#5FA8A6", "#D98C7A", ...],
  "class_keypoint_counts": [1, 5],
  "keypoint_radius": 2.0
}
```

| 字段 | 说明 |
|------|------|
| `class_names` | 类别名称列表 |
| `class_colors` | 各类别的显示颜色（按序对应，用 `#RRGGBB` 或 tkinter 颜色名） |
| `class_keypoint_counts` | 各类别的关键点数量（按序对应） |
| `keypoint_radius` | 关键点圆点半径（像素倍数），越小点越小 |

### 导出格式

保存的 `.txt` 文件为标准 YOLO Pose 格式，每行格式为：

```
class_id cx cy w h x1 y1 v1 x2 y2 v2 ...
```

- `cx, cy, w, h`：归一化目标框（center_x, center_y, width, height）
- `xi, yi, vi`：归一化关键点坐标及可见性（0=缺失, 1=遮挡, 2=可见）
- **自动补齐**：如果某类别的关键点数少于全局最大值，导出时会自动补 `0 0 0`

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+R` | 新建标注（等同于点击 Create 按钮） |
| `F8` | 撤销当前图片的最后一个标注 |
| `A` / `←` | 左移画面 |
| `D` / `→` | 右移画面 |
| `W` / `↑` | 上移画面 |
| `S` / `↓` | 下移画面 |
| 滚轮 | 缩放图片 |
| 左键拖拽 | 平移图片 |

### 依赖

- Python 3.7+
- Pillow（PIL）
- tkinter（Python 自带）
