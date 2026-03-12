/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-07-23 20:09:35
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-07-23 22:33:44
 * @FilePath: /src/planning/src/path1/path_optimization/src/path_optimization.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */

#include "path_optimization.hpp"
#include <fstream>
using namespace std;


TrajectoryOptimization::TrajectoryOptimization(std::fstream& file,ros::NodeHandle &nh) : file_(file)
{
    string line;
    // 读取txt文件
    while (getline(this->file_, line))
    {
        pcl::PointXYZI points_;
        std::istringstream iss(line);
        if (!(iss >> points_.x >> points_.y >> points_.z >> points_.intensity))
        {
            std::cout << "Error reading line: " << line << std::endl;
            continue;
        }
        points_.z = (float)points_.z;
        points_.x = (float)points_.x;
        points_.y = (float)points_.y;
        points_.intensity = (float)points_.intensity;
        opt_line_record->push_back(points_);
    }
    opt_total_points_num=opt_line_record->points.size();
    Eigen::VectorXd QPSolution(opt_total_points_num);
    QPSolution=Smooth_Reference_Line(opt_line_record);
    //原始数据文件关闭
    if (file_.is_open()) {
    file_.close();
    }

       // 打开文件进行写入
    std::ofstream outFile(optimized_FILE_NAME);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open optimized_FILE_NAME.txt" << std::endl;
        return ;
    }
    
    // 写入格式化的数据到文件
    for (int i = 0; i < opt_total_points_num*2; i=i+2) {
        outFile <<QPSolution(i) <<" "<<QPSolution(i+1)<< std::endl;
    }
    cout<<"路径优化完成！！！请检查trajectory.txt文件"<<endl;
    outFile.close();
}

/**
 * @description: 参考线平滑算法,其中可以设置x_lb、x_ub、y_ub、y_lb、w设置参数，分别为xy离原始路径的最大误差，w为各个优化权重
 * @param {Ptr} opt_line_record 待平滑原始路线
 * @return {*} 求解出的路线
 */
Eigen::VectorXd TrajectoryOptimization::Smooth_Reference_Line(pcl::PointCloud<pcl::PointXYZI>::Ptr opt_line_record)
{
    float x_lb = -0.06; // x限制值
    float x_ub = 0.06;
    float y_ub = 0.06;
    float y_lb = -0.06;

    // float w_smooth = 5;
    // float w_length = 2;
    // float w_ref = 0.000001;
    float w_smooth = 20000;
    float w_length = 20000;
    float w_ref = 0.1;
    int n_total = opt_line_record->points.size();
    cout<<"num:"<<n_total<<endl;
    Eigen::VectorXd referenceline_x = Eigen::VectorXd::Zero(n_total);
    Eigen::VectorXd referenceline_y = Eigen::VectorXd::Zero(n_total);
    for (int i = 0; i < opt_line_record->points.size(); i++)
    {
        referenceline_x(i) = opt_line_record->points[i].x;
        referenceline_y(i) = opt_line_record->points[i].y;
    }
    cout << "Path optimization progress --10% " << endl;
    cout<<" -- 参数加载中 --"<<endl;
    Eigen::SparseMatrix<double> A1(2 * n_total - 4, 2 * n_total);
    Eigen::SparseMatrix<double> A2(2 * n_total - 2, 2 * n_total);
    // 创建稀疏矩阵对象
    Eigen::SparseMatrix<double> A3(2 * n_total, 2 * n_total);
    // 遍历对角线上的元素，设置为1
    cout << "Path optimization progress --20% " << endl;
    for (int i = 0; i < 2 * n_total; ++i)
    {
        A3.insert(i, i) = 1.0; // 在(i, i)位置插入1.0
    }
    // 将所有元素插入矩阵中
    A3.makeCompressed();
    
    Eigen::MatrixXd A_cons = Eigen::MatrixXd::Identity(2 * n_total, 2 * n_total);
     cout << "Path optimization progress --35% " << endl;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * n_total, 2 * n_total);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(2 * n_total);
    
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(2 * n_total);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(2 * n_total);
    cout << "Path optimization progress --50% " << endl;
    cout<<" -- 条件赋值中 --"<<endl;
    // 限制条件赋值
    for (int i = 0; i < n_total; i++)
    {
        f(2 * i) = referenceline_x(i);
        f(2 * i + 1) = referenceline_y(i);
        lb(2 * i) = f(2 * i) + x_lb;
        ub(2 * i) = f(2 * i) + x_ub;
        lb(2 * i + 1) = f(2 * i + 1) + y_lb;
        ub(2 * i + 1) = f(2 * i + 1) + y_ub;
    }
    // A1赋值
    for (int i = 0; i < 2 * n_total - 4; i++)
    {
        A1.coeffRef(i, i) = 1;
        A1.coeffRef(i, i + 2) = -2;
        A1.coeffRef(i, i + 4) = 1;
    }
    // A2赋值
    for (int i = 0; i < 2 * n_total - 2; i++)
    {
        A2.coeffRef(i, i) = 1;
        A2.coeffRef(i, i + 2) = -1;
    }
    cout << "Path optimization progress --60% " << endl;
    cout<<" -- 求解中 --"<<endl;
    H = 2 * (w_smooth * A1.transpose() * A1 + w_length * A2.transpose() * A2 + w_ref * A3);
    f = -2 * w_ref * f;
    // osqp求解

    int NumberOfVariables = 2 * n_total;   // A矩阵的列数
    int NumberOfConstraints = 2 * n_total; // A矩阵的行数
    // 求解部分
    OsqpEigen::Solver solver;
    Eigen::SparseMatrix<double> H_osqp = H.sparseView(); // 密集矩阵转换为稀疏矩阵
    H_osqp.makeCompressed();                             // 压缩稀疏行 (CSR) 格式
    H_osqp.reserve(H.nonZeros());                        // 预分配非零元素数量
    cout << "Path optimization progress --80% " << endl;
    Eigen::SparseMatrix<double> linearMatrix = A_cons.sparseView();
    linearMatrix.makeCompressed();                 // 压缩稀疏行 (CSR) 格式
    linearMatrix.reserve(linearMatrix.nonZeros()); // 预分配非零元素数量
    // lb_osqp.setConstant(-OsqpEigen::INFTY);
    // ub_osqp.setConstant(+OsqpEigen::INFTY);
    // settings
    solver.settings()->setVerbosity(false); // 求解器信息输出控制
    // solver.settings()->setWarmStart(true); // 启用热启动
    // solver.settings()->setInitialGuessX(f); // 设置初始解向量,加速收敛
    // solver.settings()->setWarmStart(true);

    // set the initial data of the QP solver
    // 矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m
    
    if (!solver.data()->setHessianMatrix(H_osqp))
        // return 1; //设置P矩阵
        cout << "error1" << endl;
    if (!solver.data()->setGradient(f))
        // return 1; //设置q or f矩阵。当没有时设置为全0向量
        cout << "error2" << endl;
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
        // return 1; //设置线性约束的A矩阵
        cout << "error3" << endl;
    if (!solver.data()->setLowerBound(lb))
    { // return 1; //设置下边界
        cout << "error4" << endl;
    }
    if (!solver.data()->setUpperBound(ub))
    { // return 1; //设置上边界
        cout << "error5" << endl;
    }

    // instantiate the solver
    if (!solver.initSolver())
        // return 1;
        cout << "error6" << endl;
    Eigen::VectorXd QPSolution;

    // solve the QP problem

    cout << "Path optimization progress --90% " << endl;
    if (!solver.solve())
    {
        cout << "error_slove" << endl;
    }
    // get the controller input
    // clock_t time_start = clock();
    // clock_t time_end = clock();
    // time_start = clock();
    QPSolution = solver.getSolution();
    cout << "Path optimization progress --100% " << endl;
    cout << "Path Optimization Successful!" << endl;
    return QPSolution;
}



TrajectoryOptimization::~TrajectoryOptimization()
{

}
