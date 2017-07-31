#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/io/png_io.h>
#include <pcl/common/common.h>

#include <iostream>
#include <ctime>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <pcl/common/distances.h>
#include <pcl/keypoints/iss_3d.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/filters/extract_indices.h>


#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <tf/transform_datatypes.h>

using namespace Eigen;
using namespace std;
using namespace pcl;

//catkin_make -DCMAKE_BUILD_TYPE=Release


//const unsigned int MIN_CLUSTER_POINTS = 200;
const unsigned int MIN_CLUSTER_POINTS = 70;
const unsigned int MAX_CLUSTER_POINTS = 150;
//const unsigned int MIN_CLUSTER_POINTS = 200;
//const unsigned int MIN_CLUSTER_POINTS = 150;
//const unsigned int MIN_CLUSTER_POINTS = 400;
//const double CLUSTER_DISTANCE = 0.15;
const double CLUSTER_DISTANCE = 0.5;
//const double CLUSTER_DISTANCE = 0.05;
//const double CLUSTER_DISTANCE = 0.05;
//const double CLUSTER_DISTANCE = 0.5;
//const double CLUSTER_DISTANCE = 0.7;
//const double GROUND_SEGMENTATION_DISTANCE = 0.7;
const double GROUND_SEGMENTATION_DISTANCE = 1.3;

ros::Publisher pub1;
ros::Publisher pub2;

//pcl::PointCloud<pcl::PointXYZ>::Ptr globalCloud (new pcl::PointCloud<pcl::PointXYZ>);
//pcl::PointCloud<pcl::PointXYZ>::Ptr globalFeatures (new pcl::PointCloud<pcl::PointXYZ>);

void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector< pcl::PointCloud<PointXYZ> > cloudVector);

unsigned int frame=0;

template <typename PointT>
void fromPCLPointCloud2ToVelodyneCloud(const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud1D, std::vector< pcl::PointCloud<PointT> >& cloudVector, unsigned int rings)
{
  cloud1D.header   = msg.header;
  cloud1D.width    = msg.width;
  cloud1D.height   = msg.height;
  cloud1D.is_dense = msg.is_dense == 1;
  uint32_t num_points = msg.width * msg.height;
  cloud1D.points.resize (num_points);
  uint8_t* cloud_data1 = reinterpret_cast<uint8_t*>(&cloud1D.points[0]);
  
  pcl::PointCloud<PointT>* cloudPerLaser = new pcl::PointCloud<PointT>[rings];
  uint8_t* cloud_data2[rings];

  unsigned int pointsCounter[rings] = {0};

  for(unsigned int i=0; i<rings; ++i)
  {
    cloudPerLaser[i] = pcl::PointCloud<PointT>();
    cloudPerLaser[i].header   = msg.header;
    cloudPerLaser[i].width    = msg.width;
    cloudPerLaser[i].height   = msg.height;
    cloudPerLaser[i].is_dense = msg.is_dense == 1;
    cloudPerLaser[i].points.resize (num_points);
    cloud_data2[i] = reinterpret_cast<uint8_t*>(&cloudPerLaser[i].points[0]);
  }

  for (uint32_t row = 0; row < msg.height; ++row)
  {
    const uint8_t* row_data = &msg.data[row * msg.row_step];
      
    for (uint32_t col = 0; col < msg.width; ++col)
    {
        const uint8_t* msg_data = row_data + col * msg.point_step;

        //float* x = (float*)msg_data;
        //float* y = (float*)(msg_data + 4);
        //float* z = (float*)(msg_data + 8);
        //float* i = (float*)(msg_data + 16);
        uint16_t* ring = (uint16_t*)(msg_data+20);
        memcpy (cloud_data2[*ring], msg_data, 22);
        memcpy (cloud_data1, msg_data, 22);
        pointsCounter[*ring]++;
        cloud_data1 += sizeof (PointT);
        cloud_data2[*ring] += sizeof (PointT);
    }
  }

  cloudVector = std::vector< pcl::PointCloud<PointT> >(rings);

  for(unsigned int i=0; i<rings; ++i)
  {
      cloudPerLaser[i].width = pointsCounter[i];
      cloudPerLaser[i].height = 1;
      cloudPerLaser[i].points.resize (pointsCounter[i]);
      cloudVector[i] = (cloudPerLaser[i]);
  }

  delete[] cloudPerLaser;
}

void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  vector< pcl::PointCloud<pcl::PointXYZ> > cloudVector;
	pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*cloud_msg, pcl_pc2);

  fromPCLPointCloud2ToVelodyneCloud (pcl_pc2, *cloud, cloudVector, 16);

  //pcl::fromROSMsg(*cloud_msg , *cloud);
	cout<<"Frame "<<frame<<endl;
	extractFeatures(cloud, cloudVector);
	frame++;
}

void clusterize(PointCloud<PointXYZ>::Ptr cloud, vector< PointCloud<PointXYZ> >& clusters, double threshold, double minClusterPoints, double maxClusterPoints)
{
	bool isDiscontinuous[cloud->size()];
  	int numClusters = 0;

  	//for(int i = cloud->size() ; i >= 0 ; i--)
    	//if(pcl::euclideanDistance((*cloud)[i],PointXYZ(0,0,0))<1.0)
      		//cloud->erase(cloud->begin()+i);      	

	for(int i = 0 ; i < cloud->size() ; ++i)
	{
	    if(pcl::euclideanDistance((*cloud)[i],(*cloud)[(i+1)%cloud->size()])>threshold)
	    {
	      isDiscontinuous[(i+1)%cloud->size()] = true;
	      numClusters++;
	    }else{
	      isDiscontinuous[(i+1)%cloud->size()] = false;
	    }
  	}

  	int clustersArray[numClusters];
  	int n = 0;
  	//clusters = vector< PointCloud<PointXYZ> >(numClusters);

	pcl::PointCloud<pcl::PointXYZ> cloud_cluster;
  	
  	for(int i = 0; i < cloud->size(); ++i)
    	if(isDiscontinuous[i])
      		clustersArray[n++]=i;

  	//cout<<"Clusters: "<<numClusters<<endl;

  	for(int i=0; i<numClusters; ++i)
  	{
  		int pointsPerCluster = (clustersArray[(i+1)%numClusters]-clustersArray[i]+cloud->size())%cloud->size();//numero de puntos de cada cluster
   		
   		if (pointsPerCluster<minClusterPoints || pointsPerCluster>maxClusterPoints)
   			continue;
    
    	cloud_cluster.clear();
    	
    	for(int j=clustersArray[i] ; j!=clustersArray[(i+1)%numClusters] ; j=(j+1)%cloud->size())
    		cloud_cluster.push_back((*cloud)[j]);

    	clusters.push_back(cloud_cluster);
	    /*std::stringstream ss2;
	    ss2 << "velodyne_rings_clusters/object" << frame<<"_"<<ring<<"_"<<i<<".pcd";
	    pcl::io::savePCDFile<pcl::PointXYZ>(ss2.str (), cloud_cluster);*/
      	
    }
}

typedef struct linePair
{
  Eigen::VectorXf line;
  PointCloud<PointXYZ> lineCloud; 
}LinePair;

void getCorner(PointCloud<PointXYZ> in, PointCloud<PointXYZ>& corners)
{
  
  PointCloud<PointXYZ>::Ptr tmp (new PointCloud<PointXYZ>);
  copyPointCloud<PointXYZ>(in, *tmp);

  //vector<Eigen::VectorXf> linesVector;

  vector<LinePair> linePairVector;


  //cout<<tmp->points.size()<<endl;

  vector< vector<int> > lineInliersVector;

  //while(tmp->points.size()>5)
  while(tmp->points.size()>10)
  {
    
    vector<int> lineInliers;
    SampleConsensusModelLine<PointXYZ>::Ptr modelLine (new SampleConsensusModelLine<PointXYZ> (tmp));
    RandomSampleConsensus<PointXYZ> ransac (modelLine);
    ransac.setDistanceThreshold (.015);
    ransac.computeModel();
    ransac.getInliers(lineInliers);

    vector<int> samples;
    samples.push_back(lineInliers[0]);
    samples.push_back(lineInliers[lineInliers.size()-1]);

    Eigen::VectorXf line(6);

    modelLine->computeModelCoefficients (samples, line);
    modelLine->optimizeModelCoefficients (lineInliers, line, line);


    PointCloud<PointXYZ> lineCloud;
    copyPointCloud<PointXYZ>(*tmp, lineInliers, lineCloud);

    LinePair linePair;
    linePair.line = line;
    linePair.lineCloud = lineCloud;

    //linesVector.push_back(line);
    linePairVector.push_back(linePair);

    PointIndices::Ptr lineIndices(new PointIndices);
    lineIndices->indices = lineInliers;
    ExtractIndices<PointXYZ> ex;
    ex.setInputCloud (tmp);
    ex.setIndices (lineIndices);
    ex.setNegative (true);
    ex.filter (*tmp);
  }

  //cout<<linesVector[0]<<endl;

  //cout<<"size:"<<linePairVector.size()<<endl;

  int i1, i2;
  Eigen::VectorXf line[2];

  for(vector<LinePair>::iterator i=linePairVector.begin(); i!=linePairVector.end(); ++i)
  {
    for(vector<LinePair>::iterator j=linePairVector.begin(); j!=linePairVector.end(); ++j)
    {
      //if( i1!=i2 && euclideanDistance( (*i).lineCloud.back(), (*j).lineCloud.front() ) <1.0 )
      if( i1!=i2 )
      {

        Eigen::Vector3f v1((*i).line[3], (*i).line[4], (*i).line[5]);
        Eigen::Vector3f v2((*j).line[3], (*j).line[4], (*j).line[5]);
        float ang = acos(v1.dot(v2))*180/M_PI;

        //cout<<"angulo: "<<ang<<endl;
        if(ang>75 && ang<120)
        {
          line[0] = (*i).line;
          line[1] = (*j).line;

          Matrix2f A;
          Vector2f b;

          A << line[0][3], line[1][3]*(-1), line[0][4], line[1][4]*(-1);
          b << line[1][0] - line[0][0], line[1][1] - line[0][1];

          Vector2f x = A.colPivHouseholderQr().solve(b);

          float x1 = line[0][0] + line[0][3] * x[0];
          float y1 = line[0][1] + line[0][4] * x[0];
          float z1 = line[0][2] + line[0][5] * x[0];

          PointXYZ corner(x1, y1, z1);

          //cout<<euclideanDistance( corner, (*i).lineCloud.back() )<<endl;
          //cout<<euclideanDistance( corner, (*j).lineCloud.front() )<<endl;
          
          if(euclideanDistance( corner, (*i).lineCloud.back() ) <0.05 && euclideanDistance( corner, (*j).lineCloud.front()) < 0.05 )
          {
            corners.push_back(corner);
          }
        }
      }
      i2++;
    }
    i1++;
    i2 = 0;
  }
}


void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector< pcl::PointCloud<PointXYZ> > cloudVector)
{
  clock_t begin1 = clock();

  pcl::PointCloud<pcl::PointXYZ> featuresCloud;
  featuresCloud.header=cloud->header;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTmp (new pcl::PointCloud<pcl::PointXYZ>);

  int tid, nthreads, i, j, k, chunk;
  chunk = 1;
  i = 0;

  //*globalCloud += *cloud;

  /*pcl::PointCloud<pcl::PointXYZ>::Ptr* cloudTmp = new pcl::PointCloud<pcl::PointXYZ>::Ptr[16];
  vector< PointCloud<PointXYZ> >* clusters = new vector< PointCloud<PointXYZ> >[16];

  for(int c=0; c<16; c++)
	cloudTmp[c] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  #pragma omp parallel shared(nthreads,chunk) private(tid,i,j)
  {

  #pragma omp for schedule (static, chunk)*/
  for(vector< PointCloud<PointXYZ> >::iterator it = cloudVector.begin()+4 ; it != cloudVector.end(); ++it)
  //for(i = 0 ; i < 16; ++i)
  {	
  		*cloudTmp = *it;
  		//*cloudTmp[i] = cloudVector[i];
		
	    //std::stringstream ss1;
	    //ss1 << "velodyne_rings_clusters/objects" << frame<<"_"<<ring<<".pcd";
	    //pcl::io::savePCDFile<pcl::PointXYZ>(ss1.str (), *cloudTmp[i]);

	    vector< PointCloud<PointXYZ> > clusters;
	    clusterize(cloudTmp, clusters, CLUSTER_DISTANCE, MIN_CLUSTER_POINTS, MAX_CLUSTER_POINTS);
	    
	    //clusterize(cloudTmp[i], clusters[i], CLUSTER_DISTANCE);
	    j = 0;

	    //clusters
	    for (vector< PointCloud<PointXYZ> >::iterator it2 = clusters.begin() ; it2 != clusters.end(); ++it2)
	    //for (vector< PointCloud<PointXYZ> >::iterator it2 = clusters[i].begin() ; it2 != clusters[i].end(); ++it2)
	    {
	    	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
			  //*cloud_cluster = *it2;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_features (new pcl::PointCloud<pcl::PointXYZ>);
        getCorner(*it2, *cloud_features);
        
        featuresCloud += *cloud_features;

	    	j++;
	    }
	    i++;
  }
  //}

  pcl::PCLPointCloud2 cloud2;
  cloud2.header=cloud->header;
  toPCLPointCloud2(featuresCloud, cloud2);
  pub1.publish (cloud2);

  /*pcl::PCLPointCloud2 cloud2;
  cloud2.header=cloud->header;

  toPCLPointCloud2(featuresCloud, cloud2);
  pub1.publish (cloud2);

  sensor_msgs::PointCloud2 features_msg;
  pcl::toROSMsg<pcl::InterestPoint>(features,features_msg);
  pub2.publish (features_msg);*/

  clock_t end1 = clock();  
  double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
  cout<<"************************************"<<endl<<endl;
  cout<<"TIEMPO: "<<elapsed_secs1<<endl;
  cout<<"************************************"<<endl<<endl;

  /*cout<<"************************************"<<endl<<endl;
  cout<<"TIEMPO TOTAL: "<<elapsed_secs1+elapsed_secs2<<endl;
  cout<<"************************************"<<endl<<endl;*/
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "corner_extraction");
  ros::NodeHandle nh;


  cout<<"HolaMundo"<<endl;

  omp_set_num_threads(4);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub1 = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, cloud_callback);
  pub1 = nh.advertise<sensor_msgs::PointCloud2> ("/features1", 1);
  pub2 = nh.advertise<sensor_msgs::PointCloud2> ("/features2", 1);
  ros::spin ();
}