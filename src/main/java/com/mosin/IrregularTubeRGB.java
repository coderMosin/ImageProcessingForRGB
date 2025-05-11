package com.mosin;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.*;

public class IrregularTubeRGB {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String imagePath = "D:\\java-demo\\rgb1.jpg"; // 替换为实际图片路径
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("无法读取图片");
            return;
        }

        // 预处理：灰度化、高斯模糊、二值化
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new org.opencv.core.Size(7, 7), 0);

        Mat thresh = new Mat();
        Imgproc.threshold(blurred, thresh, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU); // 替换原自适应阈值 // 可能需调整阈值

        Imgcodecs.imwrite("D:\\java-demo\\thresh.jpg", thresh); // 输出二值化图像，检查效果

        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        int rowThreshold = 10; // 可以根据实际情况调整

        // 按 y 坐标分组
        Map<Integer, List<MatOfPoint>> rowContours = new HashMap<>();
        for (MatOfPoint contour : contours) {
            Rect boundingRect = Imgproc.boundingRect(contour);
            int currentY = boundingRect.y;
            boolean foundRow = false;
            for (int row : rowContours.keySet()) {
                if (Math.abs(currentY - row) <= rowThreshold) {
                    rowContours.get(row).add(contour);
                    foundRow = true;
                    break;
                }
            }
            if (!foundRow) {
                List<MatOfPoint> newRow = new ArrayList<>();
                newRow.add(contour);
                rowContours.put(currentY, newRow);
            }
        }

        // 对每组内的轮廓按 x 坐标排序
        List<Integer> sortedRows = new ArrayList<>(rowContours.keySet());
        Collections.sort(sortedRows);

        int row = 1;
        for (int currentRow : sortedRows) {
            List<MatOfPoint> currentRowContours = rowContours.get(currentRow);
            Collections.sort(currentRowContours, Comparator.comparingInt(c -> {
                Rect boundingRect = Imgproc.boundingRect(c);
                return boundingRect.x;
            }));

            int col = 1;
            for (MatOfPoint contour : currentRowContours) {
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                double area = Imgproc.contourArea(contour);
                if (area < 4) // 根据实际情况调整最小面积（如 100、200）
                    continue;

                Rect boundingRect = Imgproc.boundingRect(contour);
                int centerX = boundingRect.x + boundingRect.width / 2;
                int centerY = boundingRect.y + boundingRect.height / 2;

                int rSum = 0, gSum = 0, bSum = 0;
                int count = 0; // 统计轮廓内有效点数

                // 上下偏移 2 个像素点取平均值
                for (int yOffset = -2; yOffset <= 2; yOffset++) {
                    int y = centerY + yOffset;
                    if (y >= 0 && y < image.rows()) {
                        int x = centerX;
                        if (x >= 0 && x < image.cols()) {
                            Point pt = new Point(x, y);
                            // 检查点是否在轮廓内（dist >= 0 表示在内部或边缘）
                            double dist = Imgproc.pointPolygonTest(contour2f, pt, false);
                            if (dist >= 0) {
                                Mat pixel = image.row(y).col(x);
                                double[] bgr = pixel.get(0, 0);
                                if (bgr != null && bgr.length >= 3) {
                                    bSum += bgr[0]; // B 通道（OpenCV 中图像为 BGR 顺序）
                                    gSum += bgr[1]; // G 通道
                                    rSum += bgr[2]; // R 通道
                                    count++;
                                }
                            }
                        }
                    }
                }

                if (count == 0) {
                    col++;
                    continue;
                }

                // 计算平均 RGB
                int r = rSum / count;
                int g = gSum / count;
                int b = bSum / count;

                // 计算 R 与 B 的比值
                double ratio = (r == 0) ? 0 : (double) b / r;
                ratio = Math.round(ratio * 100.0) / 100.0;

                // 打印取的像素点坐标
                System.out.printf("第%d排第%d个Tube: 坐标 (%d, %d), RGB (%d, %d, %d), B/R Ratio: %.2f%n", row, col, centerX, centerY, r, g, b, ratio);
                col++;
            }
            row++;
        }
    }
}