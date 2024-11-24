import os
import sys
import cv2
import pytesseract
import numpy as np
from model import model

image_path = "./3.png"
scale = 100
line_width = 5
offset = 1
enable_ai = 0

def clean_and_merge_lines(lines, threshold=1):
    cleaned_lines = []
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        if line[0][0] == line[1][0]:
            vertical_lines.append(line)
        elif line[0][1] == line[1][1]:
            horizontal_lines.append(line)
        else:
            continue

    vertical_lines.sort(key=lambda l: (l[0][0], min(l[0][1], l[1][1])))
    merged_vertical_lines = []
    for line in vertical_lines:
        if not merged_vertical_lines:
            merged_vertical_lines.append(line)
        else:
            last = merged_vertical_lines[-1]
            if abs(last[0][0] - line[0][0]) <= threshold and (
                max(last[0][1], last[1][1]) >= min(line[0][1], line[1][1]) - threshold
            ):
                new_line = (
                    (last[0][0], min(last[0][1], last[1][1], line[0][1], line[1][1])),
                    (last[0][0], max(last[0][1], last[1][1], line[0][1], line[1][1])),
                )
                merged_vertical_lines[-1] = new_line
            else:
                merged_vertical_lines.append(line)

    horizontal_lines.sort(key=lambda l: (l[0][1], min(l[0][0], l[1][0])))
    merged_horizontal_lines = []
    for line in horizontal_lines:
        if not merged_horizontal_lines:
            merged_horizontal_lines.append(line)
        else:
            last = merged_horizontal_lines[-1]
            if abs(last[0][1] - line[0][1]) <= threshold and (
                max(last[0][0], last[1][0]) >= min(line[0][0], line[1][0]) - threshold
            ):
                new_line = (
                    (min(last[0][0], last[1][0], line[0][0], line[1][0]), last[0][1]),
                    (max(last[0][0], last[1][0], line[0][0], line[1][0]), last[0][1]),
                )
                merged_horizontal_lines[-1] = new_line
            else:
                merged_horizontal_lines.append(line)

    cleaned_lines.extend(merged_vertical_lines)
    cleaned_lines.extend(merged_horizontal_lines)
    return cleaned_lines

def remove_isolated_lines(lines):
    def lines_intersect(line1, line2, tolerance = offset):
        from math import isclose

        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        def cross_product(x1, y1, x2, y2, x3, y3):
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        d1 = cross_product(x1, y1, x2, y2, x3, y3)
        d2 = cross_product(x1, y1, x2, y2, x4, y4)

        d3 = cross_product(x3, y3, x4, y4, x1, y1)
        d4 = cross_product(x3, y3, x4, y4, x2, y2)

        if d1 * d2 < 0 and d3 * d4 < 0:
            return True

        def is_point_near_segment(x1, y1, x2, y2, x3, y3, tol):
            if x1 == x2 and y1 == y2:
                return abs(x3 - x1) <= tol and abs(y3 - y1) <= tol
            px = x2 - x1
            py = y2 - y1
            norm = px * px + py * py
            u = ((x3 - x1) * px + (y3 - y1) * py) / norm
            u = max(0, min(1, u))
            closest_x = x1 + u * px
            closest_y = y1 + u * py
            distance = ((closest_x - x3) ** 2 + (closest_y - y3) ** 2) ** 0.5
            return distance <= tol

        if is_point_near_segment(x1, y1, x2, y2, x3, y3, tolerance):
            return True
        if is_point_near_segment(x1, y1, x2, y2, x4, y4, tolerance):
            return True
        if is_point_near_segment(x3, y3, x4, y4, x1, y1, tolerance):
            return True
        if is_point_near_segment(x3, y3, x4, y4, x2, y2, tolerance):
            return True

        return False

    keep_lines = []
    for i, line1 in enumerate(lines):
        has_intersection = False
        for j, line2 in enumerate(lines):
            if i != j and lines_intersect(line1, line2):
                has_intersection = True
                break
        if has_intersection:
            keep_lines.append(line1)

    return keep_lines

def post_process_predictions(predictions):
    corrected_predictions = []
    for line in predictions:
        x1, y1, x2, y2 = line
        if abs(x1 - x2) < 1e-3:
            corrected_line = [x1, y1, x1, y2]
        elif abs(y1 - y2) < 1e-3:
            corrected_line = [x1, y1, x2, y1]
        else:
            if abs(x1 - x2) > abs(y1 - y2):
                corrected_line = [x1, y2, x2, y2]
            else:
                corrected_line = [x2, y1, x2, y2]
        corrected_predictions.append(corrected_line)
    return np.array(corrected_predictions)


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
h, w = image.shape
background_color = (255, 255, 255)

boxes = pytesseract.image_to_boxes(image, lang='chi_tra')
for box in boxes.splitlines():
    box = box.split(' ')
    char = box[0]
    x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
    y1 = h - y1
    y2 = h - y2
    if char == '~':
        continue
    if abs(x1 - x2) <= line_width or abs(y1 - y2) <= line_width:
        continue
    cv2.rectangle(image, (x1, y2), (x2, y1), (255, 255, 255), -1)

cv2.imwrite("./output.png", image)

# grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
# grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# edges = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
# _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
# lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=50)

edges = cv2.Canny(image, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=50)

line_coordinates = []
records = set()

def error_function(x1, y1, x2, y2, offset=2):
    offset_lines = set()
    
    for dx1 in range(-offset, offset + 1):
        for dy1 in range(-offset, offset + 1):
            for dx2 in range(-offset, offset + 1):
                for dy2 in range(-offset, offset + 1):
                    new_x1 = x1 + dx1
                    new_y1 = y1 + dy1
                    new_x2 = x2 + dx2
                    new_y2 = y2 + dy2
                    
                    offset_lines.add(((new_x1, new_y1), (new_x2, new_y2)))
    return offset_lines


if lines is not None:
    scale = round(max((np.max(lines, axis=0) / scale)[0]))
    for line in lines:
        y1, x1, y2, x2 = map(lambda x: round(int(x) / scale), line[0])
        if x1 == x2:
            if abs(y1 - y2) <= offset * 2:
                continue
        elif y1 == y2:
            if abs(x1 - x2) <= offset * 2:
                continue
        else:
            continue
        # flag = 1
        # if ((x1, y1), (x2, y2)) in records:
        #     flag = 0
        # if flag:
        #     pass
        records.update(error_function(x1, y1, x2, y2))
        line_coordinates.append(((x1, y1), (x2, y2)))
else:
    print("No line detected!")
    sys.exit(0)

line_coordinates = clean_and_merge_lines(line_coordinates)
if enable_ai:
    line_coordinates.sort()
    print(line_coordinates)
    line_coordinates = np.array([(line[0][0], line[0][1], line[1][0], line[1][1]) for line in line_coordinates], dtype=np.float32)
    line_coordinates = np.expand_dims(line_coordinates, axis=0)
    R = np.max(line_coordinates, 1)
    line_coordinates = line_coordinates / R
    line_coordinates = np.round(model.predict(line_coordinates)[0] * R)
    for (y1, x1, y2, x2) in line_coordinates:
        if x1 == x2 or y1 == y2:
            pass
        else:
            print(x1, y1, x2, y2)
    line_coordinates = post_process_predictions(line_coordinates)
    line_coordinates = [
        ((round(line[0]), round(line[1])),
         (round(line[2]), round(line[3])))
        for line in line_coordinates
    ]
    line_coordinates.sort()
print(line_coordinates, len(line_coordinates))
# line_coordinates = remove_isolated_lines(line_coordinates)

if len(line_coordinates) == 0:
    sys.exit(0)

grid_height = 2 + max(max(y1 for (y1, _), (y2, _) in line_coordinates), max(y2 for (y1, _), (y2, _) in line_coordinates)) + 1
grid_width = 2 + max(max(x1 for (_, x1), (_, x2) in line_coordinates), max(x2 for (_, x1), (_, x2) in line_coordinates)) + 1

grid_adjacent = [['empty'] + ['empty' for _ in range(grid_width)] for _ in range(grid_height)]

for (y1, x1), (y2, x2) in line_coordinates:
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            grid_adjacent[y + 1][x1 + 1] = 'line'
    elif y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            grid_adjacent[y1 + 1][x + 1] = 'line'

html_rows_adjacent = []
border_horizon = 'border-top: 1px solid black;'
border_vertical = 'border-left: 1px solid black;'
for y, row in enumerate(grid_adjacent):
    html_row = '<tr>'
    for x, cell in enumerate(row):
        if x == 0 or x >= grid_width - 1:
            continue
        if cell == 'line':
            border_styles = set()
            if grid_adjacent[y][x - 1] == 'line' and grid_adjacent[y][x + 1] == 'line':
                border_styles.add(border_horizon)
            if grid_adjacent[y - 1][x] == 'line' and grid_adjacent[y + 1][x] == 'line':
                border_styles.add(border_vertical)
            for i in [-1, 1]:
                if grid_adjacent[y][x + i] == 'line' and grid_adjacent[y + i][x] == 'line':
                    border_styles.add(border_vertical)
                    border_styles.add(border_horizon)
                if grid_adjacent[y + i][x] == 'line' and grid_adjacent[y][x + i] == 'line':
                    border_styles.add(border_vertical)
                    border_styles.add(border_horizon)
                if grid_adjacent[y][x + i] == 'line' and grid_adjacent[y - i][x] == 'line':
                    border_styles.add(border_vertical)
                    border_styles.add(border_horizon)
                if grid_adjacent[y + i][x] == 'line' and grid_adjacent[y][x - i] == 'line':
                    border_styles.add(border_vertical)
                    border_styles.add(border_horizon)
            if grid_adjacent[y][x - 1] == 'line' and grid_adjacent[y][x + 1] == 'line' and \
               grid_adjacent[y + 1][x] != 'line' and border_vertical in border_styles:
               border_styles.remove(border_vertical)
            if grid_adjacent[y + 1][x] == 'line' and grid_adjacent[y - 1][x] == 'line' and \
               grid_adjacent[y][x + 1] != 'line' and border_horizon in border_styles:
               border_styles.remove(border_horizon)
            if (grid_adjacent[y + 1][x] != 'line' and grid_adjacent[y][x + 1] != 'line') and \
               (grid_adjacent[y - 1][x] == 'line' and grid_adjacent[y][x - 1] == 'line'):
                border_styles.clear()
            if (grid_adjacent[y + 1][x] != 'line' and grid_adjacent[y][x - 1] != 'line') and \
               (grid_adjacent[y - 1][x] == 'line' and grid_adjacent[y][x + 1] == 'line'):
                if border_vertical in border_styles:
                    border_styles.remove(border_vertical)
            if (grid_adjacent[y - 1][x] != 'line' and grid_adjacent[y][x + 1] != 'line') and \
               (grid_adjacent[y + 1][x] == 'line' and grid_adjacent[y][x - 1] == 'line'):
                if border_horizon in border_styles:
                    border_styles.remove(border_horizon)


            style = ' '.join(border_styles)
            html_row += f'<td class="line" style="{style}"></td>'
        else:
            html_row += '<td><div class="text" contenteditable="true"></div></td>'
    html_row += '</tr>'
    html_rows_adjacent.append(html_row)
html_rows_adjacent.pop(0)
html_rows_adjacent.pop()

html_content_adjacent = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Representation</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 80%;
            white-space: nowrap;
            overflow: visible;  
            position: relative;
        }}
        td {{
            height: 10px;
            line-height: 1px;
            border: none;
            white-space: nowrap;
            overflow: visible;  
            line-height: normal;
        }}
        .line {{
            background-color: white;
            border: none;
        }}
        .text {{
            font-size: 30px;
            width: 100%;
            height: 100%;
        }}
        .text:empty {{
            font-size: 1px;
        }}
    </style>
</head>
<body>
    <table>
        {rows}
    </table>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const table = document.querySelector("table");

            function mergeCells() {{
                const rows = table.rows;
                for (let row of rows) {{
                    let previousCell = null;
                    let colspan = 1;

                    for (let i = 0; i < row.cells.length;) {{
                        const cell = row.cells[i];
                        const editableDiv = cell.querySelector(".text");

                        if (!editableDiv || !editableDiv.isContentEditable) {{
                            previousCell = null;
                            colspan = 1;
                            i ++;
                            continue;
                        }}

                        if (previousCell) {{
                            const prevContent = previousCell.querySelector(".text").innerText.trim();
                            const currentContent = editableDiv.innerText.trim();

                            colspan++;
                            previousCell.setAttribute("colspan", colspan);
                            previousCell.setAttribute("style", `max-width: ${{20 * colspan}}px`);
                            cell.remove();
                        }} else {{
                            previousCell = cell;
                            colspan = 1;
                            i++;
                        }}
                    }}
                }}
            }}
            mergeCells();
        }});
    </script>

</body>
</html>
""".format(rows="\n".join((html_rows_adjacent)))

output_path_adjacent = "./graph_representation_adjacent.html"
with open(output_path_adjacent, "w") as file:
    file.write(html_content_adjacent)

