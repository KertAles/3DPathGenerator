// CMakeProject1.cpp : Defines the entry point for the application.
//

#include "CMakeProject1.h"
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>

int number_selected_points = 0;
pcl::PointXYZ picked_points[4];
int picked_points_idx[4];

// callback for shift-clicking a point
void pointPickingCallback(const pcl::visualization::PointPickingEvent& event, void* viewer_void) {
    pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);

    if (event.getPointIndex() == -1 || number_selected_points >= 4)
        return;

    int idx = event.getPointIndex();
    float x, y, z;
    event.getPoint(x, y, z);
    std::cout << "Picked point " << idx << ": " << x << ", " << y << ", " << z << std::endl;

    pcl::PointXYZ picked_point(x, y, z);
    pcl::PointCloud<pcl::PointXYZ>::Ptr picked_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    picked_cloud->push_back(picked_point);

    picked_points[number_selected_points] = picked_point;
    picked_points_idx[number_selected_points] = idx;
    number_selected_points++;

    
    std::string point_id = "picked_point_" + std::to_string(idx);

    viewer->addPointCloud<pcl::PointXYZ>(picked_cloud, point_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
        1.0, 0.0, 0.0,
        point_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        10.0,
        point_id);
}


float distance(const Eigen::Vector3f pt1, const Eigen::Vector3f pt2) {
	return sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2) + pow(pt1[2] - pt2[2], 2));
}

// project point on plane based on the plane normal
Eigen::Vector3f project_point_on_plane(Eigen::Vector3f picked_pt, Eigen::Vector3f origin_point, Eigen::Vector3f normal) {
    normal.normalize();

	Eigen::Vector3f v = picked_pt - origin_point;
	float dist = v.dot(normal);

	Eigen::Vector3f pt_proj = picked_pt - dist * normal;

    return pt_proj;
}


bool isPointInTriangle(
    const Eigen::Vector3f P,
    const Eigen::Vector3f A,
    const Eigen::Vector3f B,
    const Eigen::Vector3f C)
{
	const float epsilon = 1e-2*5; // Small tolerance for floating point comparisons
    Eigen::Vector3f v0 = C - A;
    Eigen::Vector3f v1 = B - A;
    Eigen::Vector3f v2 = P - A;

    float dot00 = v0.dot(v0);
    float dot01 = v0.dot(v1);
    float dot02 = v0.dot(v2);
    float dot11 = v1.dot(v1);
    float dot12 = v1.dot(v2);

    float denom = dot00 * dot11 - dot01 * dot01;
    if (denom == 0) return false; // Degenerate triangle

    float u = (dot11 * dot02 - dot01 * dot12) / denom;
    float v = (dot00 * dot12 - dot01 * dot02) / denom;

    return (u >= -epsilon) && (v >= -epsilon) && (u + v <= (1 + epsilon));
}


// Used to determine if a point is inside the quad projected onto a plane
bool isPointInQuad(
    const Eigen::Vector3f P,
    const Eigen::Vector3f A,
    const Eigen::Vector3f B,
    const Eigen::Vector3f C,
    const Eigen::Vector3f D
) {
    return isPointInTriangle(P, A, B, C) || isPointInTriangle(P, A, C, D);
}

bool isQuadOriented(
    const Eigen::Vector3f A,
    const Eigen::Vector3f B,
    const Eigen::Vector3f C,
    const Eigen::Vector3f D,
    float normalTolerance = 0.99f
) {
    Eigen::Vector3f AB = B - A;
    Eigen::Vector3f AC = C - A;

    // project on an estimated plane
    Eigen::Vector3f normal_quad = AB.cross(AC).normalized();
    Eigen::Vector3f B_proj = project_point_on_plane(B, A, normal_quad);
    Eigen::Vector3f C_proj = project_point_on_plane(C, A, normal_quad);
    Eigen::Vector3f D_proj = project_point_on_plane(D, A, normal_quad);

    Eigen::Vector3f n1 = (B_proj - A).cross(C_proj - A).normalized();
    Eigen::Vector3f n2 = (D_proj - C_proj).cross(A - C_proj).normalized();

    float dotNormals = n1.dot(n2);

    return (dotNormals > normalTolerance);
}


void draw_line(Eigen::Vector3f A,
                Eigen::Vector3f B,
                float step_factor,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
                pcl::PointCloud<pcl::PointXYZ>::Ptr picked_cloud,
                std::vector<pcl::PointXYZINormal> *generated_points = NULL)  {

    Eigen::Vector3f step = B - A;
    step /= step_factor;
    int m = 0;
    while (distance(A, B) > 1e-2 && m < step_factor) {
        A = A + step;

        pcl::PointXYZ picked_point(A[0], A[1], A[2]);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        int K = 1;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        pcl::PointXYZ projected_point = picked_point;

        int proj_idx = -1;

        if (kdtree.nearestKSearch(picked_point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            projected_point = cloud->points[pointIdxNKNSearch[0]];
            proj_idx = pointIdxNKNSearch[0];
        }

        pcl::Normal normal = cloud_normals->points[proj_idx];
        Eigen::Vector3f p = A;
        Eigen::Vector3f s(projected_point.x, projected_point.y, projected_point.z);
        Eigen::Vector3f n(normal.normal_x, normal.normal_y, normal.normal_z);
        n.normalize();

        Eigen::Vector3f projection = p - (p - s).dot(n) * n;

        pcl::PointXYZ normal_plane_projection_point(projection[0], projection[1], projection[2]);

        picked_cloud->push_back(normal_plane_projection_point);

        n = n - projection;
        n.normalize();
        n /= 20.0;
        if (generated_points != NULL) {
            generated_points->push_back(pcl::PointXYZINormal(projection[0], projection[1], projection[2], 0, n[0], n[1], n[2]));
        }
        m++;
}}


int main() {

    // some args
    bool draw_auxiliary_data = false;
    bool draw_normals = true;
    bool draw_borders = false;
    int num_of_points_in_row = 80;
    int interrow_points = 5;
    float row_distance = 0.025;


    // Load pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("C:/Users/alesk/source/repos/CMakeProject1/CMakeProject1/data/input.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file input.pcd \n");
        return (-1);
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Picking Example"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->registerPointPickingCallback(pointPickingCallback, static_cast<void*>(viewer.get()));

    std::cout << "Shift + Click to select four points." << std::endl;

    // Wait for points to be selected
    while (number_selected_points < 4 && !viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    std::cout << "Points chosen, generating path." << std::endl;
    std::cout << "Enter the desired distance between rows (default: 0.025)" << std::endl;
    std::cin >> row_distance;
    std::cout << "Your chosen distance: " << std::to_string(row_distance) << std::endl;

    // generate normal vectors for all points in pointcloud - used in projection on the surface
    std::vector<int> indices(4);
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = picked_points_idx[i];
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.05);

    ne.compute(*cloud_normals);

    // get corner points of the selected quad
    pcl::PointXYZ point_A = cloud->points[indices[0]];
    Eigen::Vector3f A(point_A.x, point_A.y, point_A.z);
    pcl::PointXYZ point_B = cloud->points[indices[1]];
    Eigen::Vector3f B(point_B.x, point_B.y, point_B.z);
    pcl::PointXYZ point_C = cloud->points[indices[2]];
    Eigen::Vector3f C(point_C.x, point_C.y, point_C.z);
    pcl::PointXYZ point_D = cloud->points[indices[3]];
    Eigen::Vector3f D(point_D.x, point_D.y, point_D.z);

    // Ensure the quad is consistently oriented
    if (!isQuadOriented(A, B, C, D)) {
        std::swap(indices[2], indices[3]);
        Eigen::Vector3f temp = C;
        C = D;
        D = temp;
        if (!isQuadOriented(A, B, C, D)) {

            std::swap(indices[1], indices[2]);
            Eigen::Vector3f temp = B;
            B = C;
            C = temp;

            if (!isQuadOriented(A, B, C, D)) {
                std::cout << "Quad couldn't be oriented." << std::endl;
                return -1;
            }
        }

        std::cout << "Quad orientation fixed." << std::endl;
    }

    std::cout << "Quad points consistently oriented." << std::endl;

    // Draw corner normals
    for (int j = 0; j < 4; j++) {
        int idx = indices[j];

        pcl::PointXYZ point = cloud->points[idx];
        Eigen::Vector3f pt(point.x, point.y, point.z);

        pcl::Normal normal = cloud_normals->points[idx];
        Eigen::Vector3f pt_n(normal.normal_x, normal.normal_y, normal.normal_z);

        pt_n = pt_n - pt;
        pt_n.normalize();
        pt_n /= 10;

        pt_n = pt + pt_n;

        viewer->addArrow(
            pcl::PointXYZ(pt_n[0], pt_n[1], pt_n[2]), // tip
            pcl::PointXYZ(pt[0], pt[1], pt[2]),    // base
            1.0, 0.0, 0.0,                         // red
            false,                                 // not double-sided
            "arrow" + std::to_string(j)
        );
    }


    
    // project the corner points onto the A->B vector
    Eigen::Vector3f pt_step = B - A;

    Eigen::Vector3f v_unit = pt_step.normalized();
    Eigen::Vector3f op = D - A;
    float proj_len_D = op.dot(v_unit);
    Eigen::Vector3f closest_point_D = A + proj_len_D * v_unit;

    op = C - A;
    float proj_len_B = pt_step.dot(v_unit);
    float proj_len_C = op.dot(v_unit);
    Eigen::Vector3f closest_point_C = A + proj_len_C * v_unit;

    // direction in which the rows move
    Eigen::Vector3f perpendicular_vector = D - closest_point_D;

    if (draw_auxiliary_data) {
        viewer->addArrow(
            pcl::PointXYZ(D[0], D[1], D[2]),
            pcl::PointXYZ(closest_point_D[0], closest_point_D[1], closest_point_D[2]),
            0.0, 0.0, 1.0,
            false,
            "blublub"
        );
    }

    // generate a "square" that covers all the corner points - used to keep the space between the rows
    if (proj_len_D > 0 && proj_len_C > 0) {
        closest_point_D = A;
    }
    else {
        if (proj_len_C < proj_len_D) {
            closest_point_D = A + proj_len_C * v_unit;
        }
        else {
            closest_point_D = A + proj_len_D * v_unit;
        }
    }

    if (proj_len_C < proj_len_B && proj_len_D < proj_len_B) {
        closest_point_C = B;
    }
    else {
        if (proj_len_C < proj_len_D) {
            closest_point_C = A + proj_len_D * v_unit;
        }
        else {
            closest_point_C = A + proj_len_C * v_unit;
        }
    }


    // Draw borders of the quad on the surface - not accurate due to projection
    if (draw_borders) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr border_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        draw_line(A, B, 50.0, cloud, cloud_normals, border_cloud);
        draw_line(B, C, 50.0, cloud, cloud_normals, border_cloud);
        draw_line(C, D, 50.0, cloud, cloud_normals, border_cloud);
        draw_line(D, A, 50.0, cloud, cloud_normals, border_cloud);

        viewer->addPointCloud<pcl::PointXYZ>(border_cloud, "border cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
            1.0, 0.0, 0.0, "border cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            10.0, "border cloud");

        viewer->spinOnce(100);

        std::cout << "Borders drawn." << std::endl;
    }

    // Recalculate the corner points - preventative, there's some issue that influences A, B, C, D points
    point_A = cloud->points[indices[0]];
    A = Eigen::Vector3f(point_A.x, point_A.y, point_A.z);
    point_B = cloud->points[indices[1]];
    B = Eigen::Vector3f(point_B.x, point_B.y, point_B.z);
    point_C = cloud->points[indices[2]];
    C = Eigen::Vector3f(point_C.x, point_C.y, point_C.z);
    point_D = cloud->points[indices[3]];
    D = Eigen::Vector3f(point_D.x, point_D.y, point_D.z);

    Eigen::Vector3f AB = B - A;
    Eigen::Vector3f AC = C - A;

    // project on an estimated plane
    Eigen::Vector3f normal_quad = AB.cross(AC).normalized();
    B = project_point_on_plane(B, A, normal_quad);
    C = project_point_on_plane(C, A, normal_quad);
    D = project_point_on_plane(D, A, normal_quad);

    pcl::PointCloud<pcl::PointXYZ>::Ptr picked_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<pcl::PointXYZINormal> *generated_points = new std::vector<pcl::PointXYZINormal>;
    Eigen::Vector3f last_point_in_row;


    // draw the start and end points for the initial row generation
    if (draw_auxiliary_data) {
        viewer->addSphere(
            pcl::PointXYZ(closest_point_D[0], closest_point_D[1], closest_point_D[2]), // center
            0.003, // radius
            0.0, 1.0, 1.0, // red
            "sphere D"
        );

        viewer->addSphere(
            pcl::PointXYZ(closest_point_C[0], closest_point_C[1], closest_point_C[2]), // center
            0.003, // radius
            1.0, 0.0, 1.0, // red
            "sphere C"
        );
    }
    

    // draw the first row that gets used as a reference for further row generation
    int k = 0;
    pt_step = closest_point_C - closest_point_D;
    pt_step /= num_of_points_in_row;
    while (distance(closest_point_D, closest_point_C) > 1e-3) {
        closest_point_D = closest_point_D + pt_step;

        pcl::PointXYZ picked_point(closest_point_D[0], closest_point_D[1], closest_point_D[2]);

        // find the closest point on the surface
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        int K = 1;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        pcl::PointXYZ projected_point = picked_point;
        int proj_idx = -1;

        if (kdtree.nearestKSearch(picked_point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            projected_point = cloud->points[pointIdxNKNSearch[0]];
            proj_idx = pointIdxNKNSearch[0];
        }

        // use the existing point and its normal to project the point to the "actual" closest point on the surface
        pcl::Normal normal = cloud_normals->points[proj_idx];
        Eigen::Vector3f p = closest_point_D;
        Eigen::Vector3f s(projected_point.x, projected_point.y, projected_point.z);
        Eigen::Vector3f n(normal.normal_x, normal.normal_y, normal.normal_z);
        n.normalize();

        Eigen::Vector3f projection = p - (p - s).dot(n) * n;

        pcl::PointXYZ normal_plane_projection_point(projection[0], projection[1], projection[2]);

        picked_cloud->push_back(normal_plane_projection_point);

        n = n - projection;
        n.normalize();
        n /= 20.0;
        
        generated_points->push_back(pcl::PointXYZINormal(projection[0], projection[1], projection[2], 0, n[0], n[1], n[2]));

        n = projection + n;
        k++;

        // look for the last point that is inside the quad - to determine the connection between rows
        if (isPointInQuad(projection, A, B, C, D))
        {
        last_point_in_row = projection;
        }
		
    }

    std::cout << "Reference row generated." << std::endl;

    // define some hard upper limit for number of rows - the loop exits once a row has no points inside the quad
    int rows = std::ceil(perpendicular_vector.norm() / row_distance) * 2;

    perpendicular_vector.normalize();
    perpendicular_vector *= row_distance;

    Eigen::Vector3f first_point_in_curr_row;
    Eigen::Vector3f last_point_in_curr_row;
    bool connect_at_end_of_row = true;

    // preventative recalculation
    point_A = cloud->points[indices[0]];
    A = Eigen::Vector3f(point_A.x, point_A.y, point_A.z);
    point_B = cloud->points[indices[1]];
    B = Eigen::Vector3f(point_B.x, point_B.y, point_B.z);
    point_C = cloud->points[indices[2]];
    C = Eigen::Vector3f(point_C.x, point_C.y, point_C.z);
    point_D = cloud->points[indices[3]];
    D = Eigen::Vector3f(point_D.x, point_D.y, point_D.z);

    AB = B - A;
    AC = C - A;

    normal_quad = AB.cross(AC).normalized();
    B = project_point_on_plane(B, A, normal_quad);
    C = project_point_on_plane(C, A, normal_quad);
    D = project_point_on_plane(D, A, normal_quad);

    // generate rows
    for (int l = 0; l < rows; l++) {

        first_point_in_curr_row = Eigen::Vector3f(0, 0, 0);
        last_point_in_curr_row = Eigen::Vector3f(0, 0, 0);
        // generate a row by using the previous row as reference
        for (int m = 0; m < k; m++) {
            pcl::PointXYZINormal picked_point = generated_points->at(l * k + m);
            Eigen::Vector3f picked_pt(picked_point.x, picked_point.y, picked_point.z);
            Eigen::Vector3f picked_pt_n(picked_point.normal_x, picked_point.normal_y, picked_point.normal_z);
            picked_pt_n.normalize();

            // create a point in the perpendicular direction towards the other side of the quad
            // project this point onto the normal plane of the origin point
            Eigen::Vector3f new_row_pt = picked_pt + perpendicular_vector;
            Eigen::Vector3f new_row_pt_proj = new_row_pt - perpendicular_vector.dot(picked_pt_n) * picked_pt_n;
            Eigen::Vector3f new_row_step_vector = new_row_pt_proj - picked_pt;

            // ensure distance is as defined
            new_row_step_vector.normalize();
            new_row_step_vector *= row_distance;

            new_row_pt_proj = picked_pt + new_row_step_vector;

            // find the point closest to the point projected onto the normal plane
            pcl::PointXYZ picked_point_xyz = pcl::PointXYZ(new_row_pt_proj[0], new_row_pt_proj[1], new_row_pt_proj[2]);

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(cloud);

            int K = 1;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            pcl::PointXYZ projected_point = picked_point_xyz;
            int proj_idx = -1;

            if (kdtree.nearestKSearch(picked_point_xyz, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                projected_point = cloud->points[pointIdxNKNSearch[0]];
                proj_idx = pointIdxNKNSearch[0];
            }

            // project the point on plane to the normal plane of the closest point
            pcl::Normal normal = cloud_normals->points[proj_idx];
            Eigen::Vector3f p = new_row_pt_proj;
            Eigen::Vector3f s(projected_point.x, projected_point.y, projected_point.z);
            Eigen::Vector3f n(normal.normal_x, normal.normal_y, normal.normal_z);
            n.normalize();

            Eigen::Vector3f projection = p - (p - s).dot(n) * n;

            pcl::PointXYZ normal_plane_projection_point(projection[0], projection[1], projection[2]);

            picked_cloud->push_back(normal_plane_projection_point);

            // take the closest point's normal as the new normal
            n = n - projection;
            n.normalize();
            n /= 20.0;

            generated_points->push_back(pcl::PointXYZINormal(projection[0], projection[1], projection[2], 0, n[0], n[1], n[2]));
            
            n = projection + n;

            // used to determine first and last point inside the quad - to connect the rows
            if (isPointInQuad(projection, A, B, C, D))
            {
                if (first_point_in_curr_row.norm() < 1e-6) {
                    first_point_in_curr_row = projection;
                }

                last_point_in_curr_row = projection;

                if (draw_normals and m % 5 == 0) {
                    viewer->addArrow(
                        pcl::PointXYZ(n[0], n[1], n[2]), // tip
                        pcl::PointXYZ(projection[0], projection[1], projection[2]),    // base
                        0.0, 1.0, 0.0,
                        false,
                        "arrow_path" + std::to_string((l + 1) * k + m)
                    );
                }
            }
        }

        // if no points were generated in a row - time to stop
        if (first_point_in_curr_row.norm() < 1e-6) {
            std::cout << "All rows generated." << std::endl;
            break;
        }
        else { // otherwise connect the current row with the previous
            if (draw_auxiliary_data) {
                viewer->addSphere(
                    pcl::PointXYZ(last_point_in_row[0], last_point_in_row[1], last_point_in_row[2]), // center
                    0.003, // radius
                    0.0, 0.0, 1.0, // red
                    "sphere out " + std::to_string(l + 1)
                );
            }

            // alternate between connecting the start and end of the row
			if (connect_at_end_of_row) {
				connect_at_end_of_row = false;

                if (draw_auxiliary_data) {
                    viewer->addSphere(
                        pcl::PointXYZ(last_point_in_curr_row[0], last_point_in_curr_row[1], last_point_in_curr_row[2]), // center
                        0.003, // radius
                        1.0, 0.0, 0.0, // red
                        "sphere in " + std::to_string(l + 1));
                }

                draw_line(last_point_in_row, last_point_in_curr_row, interrow_points, cloud, cloud_normals, picked_cloud);


				last_point_in_row = first_point_in_curr_row;
			}
            else {
                connect_at_end_of_row = true;
                if (draw_auxiliary_data) {
                    viewer->addSphere(
                        pcl::PointXYZ(last_point_in_curr_row[0], first_point_in_curr_row[1], last_point_in_curr_row[2]), // center
                        0.003, // radius
                        1.0, 0.0, 0.0, // red
                        "sphere in " + std::to_string(l + 1));
                }
                draw_line(last_point_in_row, first_point_in_curr_row, interrow_points, cloud, cloud_normals, picked_cloud);

                last_point_in_row = last_point_in_curr_row;
			}
        }

        std::cout << "Row " << std::to_string(l+1) << " generated." << std::endl;
        std::cout << std::to_string(generated_points->size()) << std::endl;
    }
    
    // preventative recalculation
    point_A = cloud->points[indices[0]];
    A = Eigen::Vector3f(point_A.x, point_A.y, point_A.z);
    point_B = cloud->points[indices[1]];
    B = Eigen::Vector3f(point_B.x, point_B.y, point_B.z);
    point_C = cloud->points[indices[2]];
    C = Eigen::Vector3f(point_C.x, point_C.y, point_C.z);
    point_D = cloud->points[indices[3]];
    D = Eigen::Vector3f(point_D.x, point_D.y, point_D.z);

    AB = B - A;
    AC = C - A;

    normal_quad = AB.cross(AC).normalized();
    B = project_point_on_plane(B, A, normal_quad);
    C = project_point_on_plane(C, A, normal_quad);
    D = project_point_on_plane(D, A, normal_quad);

    // during generation all points in a rectangle that covers all four corner points were generated
    // now the excess points are deleted
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    for (int i = 0; i < (*picked_cloud).size(); i++)
    {
        pcl::PointXYZ pt(picked_cloud->points[i].x, picked_cloud->points[i].y, picked_cloud->points[i].z);
		Eigen::Vector3f pt_vec(pt.x, pt.y, pt.z);
		pt_vec = project_point_on_plane(pt_vec, A, normal_quad);

        if (!isPointInQuad(pt_vec, A, B, C, D))
        {
            inliers->indices.push_back(i);
        }
    }
    extract.setInputCloud(picked_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*picked_cloud);
    
    viewer->addPointCloud<pcl::PointXYZ>(picked_cloud, "path cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
        0.0, 1.0, 0.0, "path cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        10.0, "path cloud");

    viewer->spinOnce(100);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    return 0;
}