#include "TROOT.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TError.h"
#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TH3F.h"
#include "TProfile2D.h"
#include "TChain.h"
#include "TStyle.h"
#include "TString.h"
#include "TVector3.h"
#include "TCanvas.h"
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>

#define PI 3.14159
#define euler 2.71828
#define cal_factor 0.0000385

//#include "/grid/fermiapp/products/larsoft/eigen/v3_3_3/include/eigen3/Eigen/Dense" //Needed on uboonegpvm

#include "/usr/local/Cellar/eigen/3.3.4/include/eigen3/Eigen/Dense" //Needed on MACOS
using namespace std;

struct Point {
	float x;
	float y;
	float z;
	float q;
};
struct PCAResults {
	TVector3 centroid;
	pair<TVector3,TVector3> endPoints;
	float length;
	TVector3 eVals;
	vector<TVector3> eVecs;
};
struct TrkPoint{
	double c;
	double x;
	double y;
	double z;
	double q;
	double uq;
	double vq;
	double wq;
};

struct Cluster_pt{
	double rn;
	double ev;
	double cl;
	double x;
	double y;
	double z;
	double q;
	double uq;
	double vq;
	double wq;
	double px;
	double py;
	double pz;
	double ord;
	double closeness;
	double vertex;
	double sp_vertex;
	double true_vertex;
	double is_ord;
};

struct by_y { 
	bool operator()(TrkPoint const &a, TrkPoint const &b) { 
		if(a.y == b.y) return a.x > b.x;
		else return a.y > b.y;
	}
};

struct reverse_by_y { 
	bool operator()(TrkPoint const &a, TrkPoint const &b) { 
		if(a.y == b.y) return a.x < b.x;
		else return a.y < b.y;
	}
};
typedef vector<TrkPoint> track_def;
typedef vector<Point> PointCloud;
typedef vector<Cluster_pt> Cluster;
void LoadPointCloud(PointCloud &points, const track_def &ord_trk);
PCAResults DoPCA(const PointCloud &points);
double Pythagoras(double x1,double x2,double y1,double y2,double z1,double z2);
vector<double> Unit_Vec(double x1,double y1,double z1);
vector<double> Unit_Vec_NO(double x1,double x2,double y1,double y2,double z1,double z2);
double dotProdFunc(double x1,double x2,double y1,double y2,double z1,double z2);
vector<double> CrossProd(double x1,double x2,double y1,double y2,double z1,double z2);

/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////MAIN PROGRAM STARTS////////////////////////////////////////

int main(int argc, char **argv){

	int track_num = atoi(argv[2]); // -1 -> uses all tracks in filelist; Event_Num + CLuster_id (e.g. 609776 where 6097 is Event num and 76 is Cluster ID) 
	///////////////////
	//Define Parameters
	int sample_freq = 40;

	//Clustering Algorithm Parameters
	double alpha = 5; // Sphere radius
	double ext_alpha = 21.5; // Extended cone radius
	double al_print = alpha*100; // 
	unsigned ang_points = 8; // Points in window usied for ordering algorithm
	int min_points_trk = ang_points*2; // Minimum number of points usied in ordering algorithm
	double min_cone_ang = 0.97; // Minimum Cone angle; ArcCos(0.97) -> 14 deg -> 0.24 Rad
	double max_phi = acos(min_cone_ang); // cone angle in radians
	double back_ang = -0.4; // Angle of back cone 113 deg -> 1.98 Rad
	back_ang = acos(back_ang); // back angle in radians
	
	double dist_unclustered = 2.; //Distance at which the unclustered point is considered for the unclustered/clustered ratio
	double cut_ucp_ratio = 0.10; // unclustered/clustered ratio at which to cut cluster as having to many unclustered points
	double cut_bottom_dist = 2.;// CUT: Max distance from the last clustered point and the lowest Y value point

	double eps_ang = 0.314159; // epsilon_phi paramter
	double eps_dist = 1.5; // epsilon_r parameter
	double weight;

	//Michel ID parameters
	double prev_win_size = 10; // PCA window size
	double post_win_size = 10.;
	double min_costheta = -0.85; // CUT: Min angle (Cos(Theta)) made by the PC Eigenvectors
	double theta = acos(min_costheta) * 180./ PI;
	double max_costheta =  0.96; // CUT: Max angle (Cos(Theta)) made by the PC Eigenvectors

	//Reclustering parameters
	double max_dist_for_charge = 2.; 
	int true_vertex = 0.;
	
	//Physics variables
	double el_energy, mu_energy;
	double el_energy_u, el_energy_v, el_energy_w; 
	vector<double> v_q;
	vector<double> v_uq;
	vector<double> v_vq;
	vector<double> v_wq;
	bool is_el = false;
	double residual_range, track_range;
	double dQ, dx, dE;
	double duQ, dvQ, dwQ;
	double duE, dvE, dwE;
	double t_dQ, t_dE;
	double tmean_q;
	double q_win = atof(argv[3]);
	double dQdx, dEdx;
	double duQdx, duEdx, dvQdx, dvEdx, dwQdx, dwEdx;
	double t_dQdx, t_dEdx;

	///////////////////
	//DEFINE ROOT OUTPUT FILE
	TFile *f_output;
	if(track_num == -1){ // argv[2] 
			f_output = TFile::Open(Form("results_%sX%s.root",argv[3],argv[3]),"RECREATE");
	}else{
			f_output = TFile::Open(Form("results_%s_weight.root",argv[2]),"RECREATE");
	}

	//Start root objects 
	TNtuple *nt_results = new TNtuple("nt_results","nt_results","C_ord:alpha:M_ord:Cone_Cosphi:max_phi:ord_pca_pts:sel:M_sel:min_Costheta:max_theta:purity:efficiency:ver_rms:ver_mean");
	TNtuple *nt_clus_trk = 	new TNtuple("nt_clus_trk","nt_clus_trk","run_num:ev_num:cluster_id:x:y:z:q:px:py:pz:ord:closeness:vertex:sp_vertex:true_vertex:is_ordered:is_michel");
	TNtuple *nt_plane_q = 	new TNtuple("nt_plane_q","nt_plane_q","run_num:ev_num:cluster_id:x:y:z:q:uq:vq:wq:px:py:pz");
	TH1F *h_vertex = new TH1F("h_vertex","h_vertex",40,0,40);
	TH1F *h_sp_vertex = new TH1F("h_sp_vertex","h_sp_vertex",40,0,40);
	TNtuple *nt_ddx_mu = new TNtuple("nt_ddx_mu","nt_ddx_mu","run_num:ev_num:cluster_id:rr:dQdx:dEdx");
	TNtuple *nt_ddx_cl = new TNtuple("nt_ddx_cl","nt_ddx_cl","run_num:ev_num:cluster_id:rr:dQ:dx:dQdx:dEdx");

	TNtuple *nt_plane_dQdx_cl = new TNtuple("nt_plane_dQdx_cl","nt_plane_dQdx_cl","run_num:ev_num:cluster_id:rr:dQ:duQ:dvQ:dwQ:dx:duQdx:dvQdx:dwQdx");
	TNtuple *nt_plane_dEdx_cl = new TNtuple("nt_plane_dEdx_cl","nt_plane_dEdx_cl","run_num:ev_num:cluster_id:rr:dE:duE:dvE:dwE:dx:duEdx:dvEdx:dwEdx");

	TNtuple *nt_cluster_info = new TNtuple("nt_cluster_info","nt_cluster_info","run_num:ev_num:cluster_id:el_energy:el_energy_u:el_energy_v:el_energy_w:mu_energy:vertex_res:sp_vertex_res");
	
	///////////////////
	//OUTPUT COMMENTS
	cout << "Using only topological cuts with this selection." << endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//READ IN MICHEL LIST FROM CSV
	ifstream csv_infile("Michel_candidates_vertex_v10.csv");
	vector<string> TrackData;
	std::vector<std::vector<double> > Michel_candidates;
	std::string mline;
	while (getline(csv_infile, mline,'\n')){
		TrackData.push_back(mline); //Get each line of the file as a string
	}
	int s = TrackData.size();
	for (unsigned int i=1; i<s; ++i){
		std::vector<double> v_michel;
		std::size_t first_comma = TrackData[i].find(",");	// position of the end of the name of each one in the respective string
		std::size_t second_comma = TrackData[i].find(",", first_comma + 1);
		std::size_t third_comma = TrackData[i].find(",", second_comma + 1);
		std::size_t fourth_comma = TrackData[i].find(",", third_comma + 1);
		std::size_t fifth_comma = TrackData[i].find(",", fourth_comma + 1);
		double mrun = std::stod(TrackData[i].substr(0,TrackData[i].size()));
		double meve = std::stod(TrackData[i].substr(first_comma+1,TrackData[i].size()));
		double mtrk = std::stod(TrackData[i].substr(second_comma+1,TrackData[i].size()));
		double mvrX = std::stod(TrackData[i].substr(third_comma+1,TrackData[i].size()));
		double mvrY = std::stod(TrackData[i].substr(fourth_comma+1,TrackData[i].size()));
		double mvrZ = std::stod(TrackData[i].substr(fifth_comma+1,TrackData[i].size()));
		v_michel.push_back(mrun);
		v_michel.push_back(meve);
		v_michel.push_back(mtrk);
		v_michel.push_back(mvrX);
		v_michel.push_back(mvrY);
		v_michel.push_back(mvrZ);
		Michel_candidates.push_back(v_michel);
	}
	int mcand_size = Michel_candidates.size();
	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////
	// COUNTERS

	int small_tracks = 0;
	int total_num_tracks = 0;
	int tracks_survived_ord_alg = 0;
	int michels_survived_ord_alg = 0;
	int michel_count = 0;
	int track_selected_as_michel = 0;
	int eigenval_cut = 0;
	int ord_alg_cutouts = 0;
	int ord_alg_cutouts_michels = 0;
	int sample_counter = 0;
	int pca_count = 0;
	int true_ver_count = 0;
		 	
	///////////////////
	// STUDY OUTPUTS

	double purity = 0, efficiency = 0, ver_rms = 0., ver_mean = 0., ver_sd = 0.;
	double sp_ver_rms = 0., sp_ver_mean = 0., sp_ver_sd = 0.;
	std::vector<double> v_vertex_res, v_sp_vertex_res;

	///////////////////
	// START OF FILELIST READ-IN LOOP

	std::string line;
	std::ifstream ifs(argv[1]);	
	while(std::getline(ifs, line)){
		// ROOT Settings
		gROOT->Reset();
		gErrorIgnoreLevel = kError;
		//Begin reading ROOT file
		TString filename;
		filename.Form("%s",line.c_str());	
		TFile *infile = new TFile(filename);

		////////////////////////////////////////////////////////////////////////////
		// EXTRACT EVENT METADATA
		TTree *Trun = (TTree*)infile->Get("Trun");
		Int_t run_num, ev_num;
		Trun->SetBranchAddress("runNo",&run_num);
		Trun->SetBranchAddress("eventNo",&ev_num);
		Trun->GetEntry(0);

		////////////////////////////////////////////////////////////////////////////
		// EXTRACT POINT INFORMATION (X,Y,Z,Q)
		TTree *T_charge_cluster = (TTree*)infile->Get("T_charge_cluster_nfc"); 
		Double_t cluster_id;
		Double_t qx;
		Double_t qy;
		Double_t qz;
		Double_t qc;
		Double_t uqc;
		Double_t vqc;
		Double_t wqc;

		T_charge_cluster->SetBranchAddress("qx",&qx);
		T_charge_cluster->SetBranchStatus("qx", kTRUE);
		T_charge_cluster->SetBranchAddress("qy",&qy);
		T_charge_cluster->SetBranchStatus("qy", kTRUE);
		T_charge_cluster->SetBranchAddress("qz",&qz);
		T_charge_cluster->SetBranchStatus("qz", kTRUE);
		T_charge_cluster->SetBranchAddress("qc",&qc);
		T_charge_cluster->SetBranchStatus("qc", kTRUE);
/////////////////////////////////////
		T_charge_cluster->SetBranchAddress("uq",&uqc);
		T_charge_cluster->SetBranchStatus("uq", kTRUE);
		T_charge_cluster->SetBranchAddress("vq",&uqc);
		T_charge_cluster->SetBranchStatus("vq", kTRUE);
		T_charge_cluster->SetBranchAddress("vq",&vqc);
		T_charge_cluster->SetBranchStatus("wq", kTRUE);
		T_charge_cluster->SetBranchAddress("wq",&wqc);
/////////////////////////////////////
		T_charge_cluster->SetBranchAddress("cluster_id", &cluster_id);
		T_charge_cluster->SetBranchStatus("cluster_id", kTRUE);
		int all_entries = T_charge_cluster->GetEntries();
		////////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////////
		//EXTRACT CLUSTERS FROM EVENT
		std::vector<Int_t> clusters;
		Int_t prev_cval;
		for (int i = 0; i < all_entries; ++i){
			T_charge_cluster -> GetEntry(i);
			if (i == 0){
				clusters.push_back(cluster_id);
				prev_cval = cluster_id;
			}else if(prev_cval != cluster_id){
				prev_cval = cluster_id;
				clusters.push_back(cluster_id);
			}
		}
		int num_clusters = clusters.size();
		////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////
		//LOOP THROUGH WIRE-CELL (WC) CLUSTERS IN EVENT
		for (int c = 0; c < num_clusters; ++c){
			int cluster = clusters[c]; // Obtain cluster number 
			total_num_tracks += 1; // cluster counter
			cout << "Looking at Run: " << run_num << ", Event: " << ev_num << ", Cluster: " << cluster << endl;
			//BOOLEAN PARAMETERS
			bool is_michel = false, is_selected = false, is_ordered = false, is_sp_selected = false;	
			// obtain cluster ID for single cluster study. This is compared to argv[2]
			string clus_id = to_string(ev_num) + to_string(cluster);
			int int_clus_id= stoi(clus_id);
			//Determine if cluster is a Michel from hand scan .csv db
			for (int cand = 0; cand < mcand_size; ++cand){
				if(Michel_candidates[cand][0] == run_num && Michel_candidates[cand][1] == ev_num && Michel_candidates[cand][2] == cluster){
					is_michel = true;
					break;
				}
			}
			////////////////////////////////////////////////////////////////
					
			
			////////////////////////////////////////////////////////////////
			// LOAD WC CLUSTER INFORMATION (x,y,z,q)
			track_def trk; // start struct
			//Load every point (cluster id, x, y, z, charge) of a cluster into the trk object.
			for (int i = 0; i < all_entries; ++i){
				T_charge_cluster -> GetEntry(i);
				if(cluster_id != cluster) continue; //Will only store information for current cluster
				if(track_num != -1){ // 
					if(int_clus_id != track_num) continue;
				}
				TrkPoint tempPoint;
				tempPoint.c = cluster_id;
				tempPoint.x = qx;
				tempPoint.y = qy;
				tempPoint.z = qz;
				tempPoint.q = qc;
				tempPoint.uq = uqc;
				tempPoint.vq = vqc;
				tempPoint.wq = wqc;
				trk.push_back(tempPoint);
			}
			////////////////////////////////////////////////////////////////
			// SMALL WC CLUSTER CUT
			if(trk.size() < min_points_trk + 1){//CUT: Track size has to be larger than the moving window size
				small_tracks += 1;
				continue; // #CUT
			}

			/////////////////////////////////////////////////////////////////////
			//////////////////CLUSTERING ALGORITHM BEGINS////////////////////////
			/////////////////////////////////////////////////////////////////////

			//Sort track in descending y value
			std::sort(trk.begin(), trk.end(), by_y());
			int trk_size = trk.size();
			track_def points_left; // points yet to be clustered
			track_def points_gd; // points left out of cluster
			track_def ord_trk; // points clustered
			ord_trk.push_back(trk[0]); //Highest y value point is the first point in the oredered track
			
			//Store points being tested by clustering algorithm
			for (int i = 1; i < trk_size; ++i){
				TrkPoint tempPoint;
				tempPoint.c = trk[i].c;
				tempPoint.x = trk[i].x;
				tempPoint.y = trk[i].y;
				tempPoint.z = trk[i].z;
				tempPoint.q = trk[i].q;
				tempPoint.uq = trk[i].uq;
				tempPoint.vq = trk[i].vq;
				tempPoint.wq = trk[i].wq;
				points_left.push_back(tempPoint);
			}
			int pl_size = points_left.size();
			std::sort(points_left.begin(), points_left.end(), by_y());

			//Clustering algorithm parameters
			double old_dist = 10000000.;
			int hi_weight_at = -1;
			double dist;
			double low_ord_y = 10000000.;
			double old_weight  = -1.;
			double vertex_res,sp_vertex_res;
			double num_close_unclustered = 0.;
			double bottom_dist;
			double shortest_dist = 1000000;

			//Variable needed by algorithm
			track_def ang_chunk;
			std::vector<double> vec_pca_ang;
			std::vector<double> vec_ob_temp;
			double pca_dotP;
			std::vector<double> pca_vec;
			std::vector<double> cone_vec;
			double cone_dotP, dotP_low_dist;
			double flip;
			TrkPoint VertexPoint;
			TrkPoint sp_VertexPoint;
			double far_ucp; 
			double far_ucp_ratio; 

			//Boolean parameters
			bool cone_test_fail = true;
			bool ongoing_cone_test = true;
			bool closest_point_found = true;
			bool closest_point_found_cone = false;
			bool closest_point_found_volume = false;
			bool closest_point_start_found = false;
			bool print_vals = false;
			bool closest_point_start_backcone = false;
			bool close_ucp_found = false;
			bool closest_point_found_short = false;
			bool true_vertex_found = false;

			//Start of clustering
			while(pl_size != 0){
				ang_chunk.clear();
				vec_pca_ang.clear();
				hi_weight_at = -1;
				old_weight = -1.;

				if(ord_trk.size() > ang_points){// PCA will be used when the first ang_points are clustered
					for(unsigned p = ord_trk.size() - ang_points; p < ord_trk.size(); ++p){ // load points for PCA
						TrkPoint ang_tempPoint;
						ang_tempPoint.c = ord_trk[p].c;
						ang_tempPoint.x = ord_trk[p].x;
						ang_tempPoint.y = ord_trk[p].y;
						ang_tempPoint.z = ord_trk[p].z;
						ang_tempPoint.q = ord_trk[p].q;
						ang_tempPoint.uq = ord_trk[p].uq;
						ang_tempPoint.vq = ord_trk[p].vq;
						ang_tempPoint.wq = ord_trk[p].wq;
						ang_chunk.push_back(ang_tempPoint);	
					}

					PointCloud ang_pointcloud;
					PCAResults ang_results;
					LoadPointCloud(ang_pointcloud, ang_chunk); 
					ang_results = DoPCA(ang_pointcloud); // Do PCA
					
					// Load PC eigenvector (unit vector)
					vec_pca_ang.push_back(ang_results.eVecs[0](0));
					vec_pca_ang.push_back(ang_results.eVecs[0](1));
					vec_pca_ang.push_back(ang_results.eVecs[0](2));

					//Make unit vector from first and last point used in PCA calculations := pca_vec
					pca_vec = Unit_Vec_NO(ang_chunk.at(0).x,ang_chunk.back().x,ang_chunk.at(0).y,ang_chunk.back().y,ang_chunk.at(0).z,ang_chunk.back().z);
					
					//Dot Product of PC eigenvector and pca_vec.
					pca_dotP = dotProdFunc(pca_vec[0],vec_pca_ang[0],pca_vec[1],vec_pca_ang[1],pca_vec[2],vec_pca_ang[2]);

					//Decide the orientation of eigenvector to be in direction of pca_vec
					if (pca_dotP > -pca_dotP){
						flip = 1.;
					}else{
						flip = -1.;
					}
					//flip -> flip = -1. ; no flip -> flip = -1.
					//Flip eigenvector
					vec_pca_ang[0] = flip * vec_pca_ang[0];
					vec_pca_ang[1] = flip * vec_pca_ang[1];
					vec_pca_ang[2] = flip * vec_pca_ang[2];
				}else{
					pca_vec.push_back(0);
					pca_vec.push_back(0);
					pca_vec.push_back(0);
					vec_pca_ang.push_back(0);
					vec_pca_ang.push_back(0);
					vec_pca_ang.push_back(0);
				}

				//Boolean reset
				cone_test_fail = true;
				ongoing_cone_test = true;
				closest_point_found_cone = false;
				closest_point_found_volume = false;
				closest_point_start_found = false;
				closest_point_start_backcone = false;
				closest_point_found = false;
				closest_point_found_short = false;

				dotP_low_dist = 0;

				//Cluster first ang_points based on distance alone
				if (ord_trk.size() <= ang_points){
					for (int j = 0; j < pl_size; ++j){
						cone_vec = Unit_Vec_NO(ord_trk.back().x,points_left[j].x,ord_trk.back().y,points_left[j].y,ord_trk.back().z,points_left[j].z);
						cone_dotP = dotProdFunc(cone_vec[0],vec_pca_ang[0],cone_vec[1],vec_pca_ang[1],cone_vec[2],vec_pca_ang[2]);
						cone_dotP = acos(cone_dotP);
						dist = Pythagoras(points_left[j].x,ord_trk.back().x,points_left[j].y,ord_trk.back().y,points_left[j].z,ord_trk.back().z);
						weight = pow(euler, -dist/eps_dist) * pow(euler, -cone_dotP/eps_ang);
						if(weight > old_weight){
							old_weight = weight;
							hi_weight_at = j;
							closest_point_start_found = true;
							closest_point_found = true;
						}
					}
				}
				//Priority #1: find point in cone of radius
				if (ord_trk.size() > ang_points){	
					for (int j = 0; j < pl_size; ++j){
						cone_vec = Unit_Vec_NO(ord_trk.back().x,points_left[j].x,ord_trk.back().y,points_left[j].y,ord_trk.back().z,points_left[j].z);
						cone_dotP = dotProdFunc(cone_vec[0],vec_pca_ang[0],cone_vec[1],vec_pca_ang[1],cone_vec[2],vec_pca_ang[2]);
						cone_dotP = acos(cone_dotP);
						dist = Pythagoras(points_left[j].x,ord_trk.back().x,points_left[j].y,ord_trk.back().y,points_left[j].z,ord_trk.back().z);
						weight = pow(euler, -dist/eps_dist) * pow(euler, -cone_dotP/eps_ang);
						if(cone_dotP < max_phi){
							if(dist < ext_alpha){
								if(weight > old_weight){
									old_weight = weight;
									dotP_low_dist = cone_dotP;
									hi_weight_at = j;
									closest_point_found_cone = true;
									closest_point_found = true;
								}
							}
						}
					}
				}
				//Priority #2: find point in sphere of radius alpha, excluding back cone
				if (closest_point_found == false && ord_trk.size() > ang_points){
					for (int j = 0; j < pl_size; ++j){
						cone_vec = Unit_Vec_NO(ord_trk.back().x,points_left[j].x,ord_trk.back().y,points_left[j].y,ord_trk.back().z,points_left[j].z);
						cone_dotP = dotProdFunc(cone_vec[0],vec_pca_ang[0],cone_vec[1],vec_pca_ang[1],cone_vec[2],vec_pca_ang[2]);
						cone_dotP = acos(cone_dotP);
						dist = Pythagoras(points_left[j].x,ord_trk.back().x,points_left[j].y,ord_trk.back().y,points_left[j].z,ord_trk.back().z);
						weight = pow(euler, -dist/eps_dist) * pow(euler, -cone_dotP/eps_ang);
						if(dist < alpha){
							if(cone_dotP < back_ang){	
								if(weight > old_weight){
									old_weight = weight;
									hi_weight_at = j;
									closest_point_found_volume = true;
									closest_point_found = true;
								}
							}
						}
					}
				}
				//find point with highest weight. DOES NOT CLUSTER THIS POINT
				if(closest_point_found == false || ord_trk.size() > ang_points){
					for (int j = 0; j < pl_size; ++j){
						cone_vec = Unit_Vec_NO(ord_trk.back().x,points_left[j].x,ord_trk.back().y,points_left[j].y,ord_trk.back().z,points_left[j].z);
						cone_dotP = dotProdFunc(cone_vec[0],vec_pca_ang[0],cone_vec[1],vec_pca_ang[1],cone_vec[2],vec_pca_ang[2]);
						cone_dotP = acos(cone_dotP);
						dist = Pythagoras(points_left[j].x,ord_trk.back().x,points_left[j].y,ord_trk.back().y,points_left[j].z,ord_trk.back().z);
						weight = pow(euler, -dist/eps_dist) * pow(euler, -cone_dotP/eps_ang);
						if(isnan(weight)) weight = 0.;	
						if (weight > old_weight){
							old_weight = weight;
							old_dist = dist;
							hi_weight_at = j;
						}
					}
				}
				// Store point with highest weight
				TrkPoint tempPoint;
				tempPoint.c = points_left[hi_weight_at].c;
				tempPoint.x = points_left[hi_weight_at].x;
				tempPoint.y = points_left[hi_weight_at].y;
				tempPoint.z = points_left[hi_weight_at].z;
				tempPoint.q = points_left[hi_weight_at].q;
				tempPoint.uq = points_left[hi_weight_at].uq;
				tempPoint.vq = points_left[hi_weight_at].vq;
				tempPoint.wq = points_left[hi_weight_at].wq;
				if (closest_point_found){// if closest point found, store in ord_trk object
					ord_trk.push_back(tempPoint);
					if (tempPoint.y < low_ord_y){
						low_ord_y = tempPoint.y;
					}
					old_dist = 10000000;
					old_weight = -1.;
					points_left.erase(points_left.begin() + hi_weight_at);
					pl_size = points_left.size();
				}else{ // else, store for future reclustering
					points_gd.push_back(tempPoint);
					old_dist = 10000000;
					old_weight = -1.;
					points_left.erase (points_left.begin() + hi_weight_at);
					pl_size = points_left.size();
				}
				if (pl_size == 0) break; // end loop if there are no more points to cluster
			}
			// If distance between lowest y value of unordered track and 
			// lowest y value of ordered track is greater than 10 cm.
			bottom_dist = abs(trk.back().y - low_ord_y);

			/////////////////////////////////////////////////////////////
			close_ucp_found = false;
			for (unsigned i = 0; i < points_gd.size(); ++i) {
				close_ucp_found = false;
				shortest_dist = 1000000;
				for (unsigned j = 0; j < ord_trk.size(); ++j) {
					dist = Pythagoras(points_gd.at(i).x,ord_trk.at(j).x,points_gd.at(i).y,ord_trk.at(j).y,points_gd.at(i).z,ord_trk.at(j).z);
					if(dist < shortest_dist){
						shortest_dist = dist;
					}
				}
				if(shortest_dist > dist_unclustered) num_close_unclustered += 1.;
			}
			far_ucp = num_close_unclustered;
			far_ucp_ratio = far_ucp / (ord_trk.size() + points_gd.size());
			if (far_ucp_ratio > cut_ucp_ratio){
				if(is_michel) ord_alg_cutouts_michels += 1;
			//}else if(bottom_dist > cut_bottom_dist){
			//	if(is_michel) ord_alg_cutouts_michels += 1;
			}else{
				is_ordered = true;
				if(is_michel) michels_survived_ord_alg += 1;
			}
			if(is_ordered) tracks_survived_ord_alg += 1;
			// FINISHED CLUSTERING POINTS
			/////////////////////////////////////////////////////////
			

			/////////////////////////////////////////////////////////////////////
			////////////////VERTEX FINDING ALGORITHM BEGINS//////////////////////
			/////////////////////////////////////////////////////////////////////
			
			if(is_ordered){
				//////////////////////////
				//Starting Moving Windo

				double dotProd, min_ang = -360.;
				double ev_lowest = 100000;
				double min_prod = 1000000000;
				int vertex;

				if(ord_trk.size() < min_points_trk*2) continue; // Min points cut

				for (int i = prev_win_size + 1; i < ord_trk.size() - post_win_size - 1; ++i){
					//////////////////////////
					//Define PCA window variables
					track_def prev_chunk;
					track_def post_chunk;
					PointCloud prev_points;
					PointCloud post_points;	
					PCAResults prev_results;
					PCAResults post_results;
					double prev_dotP_decide, post_dotP_decide;
					double new_prev_decide, new_post_decide;
					std::vector<double> prev_win_vec, post_win_vec;

					//Load first window with preceeding points to i
					for (int j = i - prev_win_size; j < i; ++j){ 
						TrkPoint prev_tempPoint;
						prev_tempPoint.c = ord_trk[j].c;
						prev_tempPoint.x = ord_trk[j].x;
						prev_tempPoint.y = ord_trk[j].y;
						prev_tempPoint.z = ord_trk[j].z;
						prev_tempPoint.q = ord_trk[j].q;
						prev_tempPoint.uq = ord_trk[j].uq;
						prev_tempPoint.vq = ord_trk[j].vq;
						prev_tempPoint.wq = ord_trk[j].wq;
						prev_chunk.push_back(prev_tempPoint);	
					}
					//Load second window with points following i
					for (int j = i + 1; j < i + post_win_size + 1; ++j){
						TrkPoint post_tempPoint;
						post_tempPoint.c = ord_trk[j].c;
						post_tempPoint.x = ord_trk[j].x;
						post_tempPoint.y = ord_trk[j].y;
						post_tempPoint.z = ord_trk[j].z;
						post_tempPoint.q = ord_trk[j].q;
						post_tempPoint.uq = ord_trk[j].uq;
						post_tempPoint.vq = ord_trk[j].vq;
						post_tempPoint.wq = ord_trk[j].wq;
						post_chunk.push_back(post_tempPoint);
					}
					//Do PCA
					LoadPointCloud(prev_points, prev_chunk);
					LoadPointCloud(post_points, post_chunk);
					prev_results = DoPCA(prev_points);
					post_results = DoPCA(post_points);

					//Decide on orientation of first window eigenvector
					prev_win_vec = Unit_Vec_NO(prev_chunk.back().x,prev_chunk.at(0).x,prev_chunk.back().y,prev_chunk.at(0).y,prev_chunk.back().z,prev_chunk.at(0).z);
					prev_dotP_decide = dotProdFunc(prev_results.eVecs[0](0),prev_win_vec[0],prev_results.eVecs[0](1),prev_win_vec[1],prev_results.eVecs[0](2),prev_win_vec[2]);			 		
					if(prev_dotP_decide > -prev_dotP_decide){
						new_prev_decide = -1.;
					}else{
						new_prev_decide = 1.;
					}
					prev_results.eVecs[0] = new_prev_decide*prev_results.eVecs[0];

					//Decide on orientation of second window eigenvector
					post_win_vec = Unit_Vec_NO(post_chunk.back().x,post_chunk.at(0).x,post_chunk.back().y,post_chunk.at(0).y,post_chunk.back().z,post_chunk.at(0).z);
					post_dotP_decide = dotProdFunc(post_results.eVecs[0](0),post_win_vec[0],post_results.eVecs[0](1),post_win_vec[1],post_results.eVecs[0](2),post_win_vec[2]);
					if(post_dotP_decide > -post_dotP_decide){
						new_post_decide = 1.;
					}else{
						new_post_decide = -1.;
					}
					post_results.eVecs[0] = new_post_decide*post_results.eVecs[0];

					//Take dot product to calculate angle
					dotProd = dotProdFunc(prev_results.eVecs[0](0),post_results.eVecs[0](0),prev_results.eVecs[0](1),post_results.eVecs[0](1),prev_results.eVecs[0](2),post_results.eVecs[0](2));
					
					if(dotProd > min_ang){ // find point with lowest angle and pass on coordinates and charge
						min_ang = dotProd;	
						vertex = i;
						VertexPoint.c = ord_trk[vertex].c;
						VertexPoint.x = ord_trk[vertex].x;
						VertexPoint.y = ord_trk[vertex].y;
						VertexPoint.z = ord_trk[vertex].z;
						VertexPoint.q = ord_trk[vertex].c;

					}
				}
				//////////////////////////////////////////////////////////////////////
				double dist_low_vert_y;
				dist_low_vert_y = abs(VertexPoint.y - low_ord_y);
				
				if(dist_low_vert_y > 15) continue; //Michel electron cutoff distance based on energy spectrum
				//angle cuts
				if(min_ang < min_costheta) continue; 
				if(min_ang > max_costheta) continue;
				track_selected_as_michel += 1;
				is_selected = true;
				if(is_michel){ 
					michel_count += 1;
					for (int cand = 0; cand < mcand_size; ++cand){
						if(Michel_candidates[cand][0] == run_num && Michel_candidates[cand][1] == ev_num && Michel_candidates[cand][2] == cluster){
							if(Michel_candidates[cand][3] == 0 && Michel_candidates[cand][4] == 0 && Michel_candidates[cand][5] == 0) break;
							vertex_res = Pythagoras(Michel_candidates[cand][3],VertexPoint.x,Michel_candidates[cand][4],VertexPoint.y,Michel_candidates[cand][5],VertexPoint.z);
							h_vertex -> Fill(vertex_res);
							ver_rms += pow(vertex_res, 2.);
							ver_mean += vertex_res;
							v_vertex_res.push_back(vertex_res);			
							break;
						}
					}
				}
			}

			//////////////////////////////
			// SECOND PASS VERTEX CALCULATION ON SELECTED EVENTS
			double sp_prev_win_size = 9;
			double sp_post_win_size = 9;
			double max_win_size = 10;
			vector<double> v_sp_win_size;
			
			if(is_selected){
				//////////////////////////
				//Starting Moving Window

				double dotProd, min_ang = -360.;
				double ev_lowest = 100000;
				double min_prod = 1000000000;
				int vertex;

				if(ord_trk.size() < min_points_trk*2) continue;
				for (int i = sp_prev_win_size + 1; i < ord_trk.size() - sp_post_win_size - 1; ++i){
					track_def prev_chunk;
					track_def post_chunk;
					PointCloud prev_points;
					PointCloud post_points;	
					PCAResults prev_results;
					PCAResults post_results;
					TrkPoint prev_first_point;
					TrkPoint post_last_point;
					double prev_dotP_decide, post_dotP_decide;
					double new_prev_decide, new_post_decide;
					std::vector<double> prev_win_vec, post_win_vec;
					
					for (int j = i - sp_prev_win_size; j < i; ++j){ // #prevwincalc
						TrkPoint prev_tempPoint;
						prev_tempPoint.c = ord_trk[j].c;
						prev_tempPoint.x = ord_trk[j].x;
						prev_tempPoint.y = ord_trk[j].y;
						prev_tempPoint.z = ord_trk[j].z;
						prev_tempPoint.q = ord_trk[j].q;
						prev_tempPoint.uq = ord_trk[j].uq;
						prev_tempPoint.vq = ord_trk[j].vq;
						prev_tempPoint.wq = ord_trk[j].wq;
						prev_chunk.push_back(prev_tempPoint);	
					}

					for (int j = i + 1; j < i + sp_post_win_size + 1; ++j){// #postwincalc
						TrkPoint post_tempPoint;
						post_tempPoint.c = ord_trk[j].c;
						post_tempPoint.x = ord_trk[j].x;
						post_tempPoint.y = ord_trk[j].y;
						post_tempPoint.z = ord_trk[j].z;
						post_tempPoint.q = ord_trk[j].q;
						post_tempPoint.uq = ord_trk[j].uq;
						post_tempPoint.vq = ord_trk[j].vq;
						post_tempPoint.wq = ord_trk[j].wq;
						post_chunk.push_back(post_tempPoint);
					}

					LoadPointCloud(prev_points, prev_chunk);
					LoadPointCloud(post_points, post_chunk);
					prev_results = DoPCA(prev_points);
					post_results = DoPCA(post_points);

					prev_win_vec = Unit_Vec_NO(prev_chunk.back().x,prev_chunk.at(0).x,prev_chunk.back().y,prev_chunk.at(0).y,prev_chunk.back().z,prev_chunk.at(0).z);
					prev_dotP_decide = dotProdFunc(prev_results.eVecs[0](0),prev_win_vec[0],prev_results.eVecs[0](1),prev_win_vec[1],prev_results.eVecs[0](2),prev_win_vec[2]);			 		
					if(prev_dotP_decide > -prev_dotP_decide){
						new_prev_decide = -1.;
					}else{
						new_prev_decide = 1.;
					}
					post_win_vec = Unit_Vec_NO(post_chunk.back().x,post_chunk.at(0).x,post_chunk.back().y,post_chunk.at(0).y,post_chunk.back().z,post_chunk.at(0).z);
					post_dotP_decide = dotProdFunc(post_results.eVecs[0](0),post_win_vec[0],post_results.eVecs[0](1),post_win_vec[1],post_results.eVecs[0](2),post_win_vec[2]);
					if(post_dotP_decide > -post_dotP_decide){
						new_post_decide = 1.;
					}else{
						new_post_decide = -1.;
					}
					prev_results.eVecs[0] = new_prev_decide*prev_results.eVecs[0];
					post_results.eVecs[0] = new_post_decide*post_results.eVecs[0];

					dotProd = dotProdFunc(prev_results.eVecs[0](0),post_results.eVecs[0](0),prev_results.eVecs[0](1),post_results.eVecs[0](1),prev_results.eVecs[0](2),post_results.eVecs[0](2));
					
					if(dotProd < min_costheta) continue;
					if(dotProd > max_costheta) continue;

					if(dotProd > min_ang){
						min_ang = dotProd;	
						vertex = i;
						sp_VertexPoint.c = ord_trk[vertex].c;
						sp_VertexPoint.x = ord_trk[vertex].x;
						sp_VertexPoint.y = ord_trk[vertex].y;
						sp_VertexPoint.z = ord_trk[vertex].z;
						sp_VertexPoint.q = ord_trk[vertex].c;
					}
				}
				//////////////////////////////////////////////////////////////////////
				if(is_michel){//Second Pass Calculation vertex resolution calculation
					for (int cand = 0; cand < mcand_size; ++cand){
						if(Michel_candidates[cand][0] == run_num && Michel_candidates[cand][1] == ev_num && Michel_candidates[cand][2] == cluster){
							if(Michel_candidates[cand][3] == 0 && Michel_candidates[cand][4] == 0 && Michel_candidates[cand][5] == 0) break;
							sp_vertex_res = Pythagoras(Michel_candidates[cand][3],sp_VertexPoint.x,Michel_candidates[cand][4],sp_VertexPoint.y,Michel_candidates[cand][5],sp_VertexPoint.z);
							h_sp_vertex -> Fill(sp_vertex_res);
							sp_ver_rms += pow(sp_vertex_res, 2.);
							sp_ver_mean += sp_vertex_res;
							v_sp_vertex_res.push_back(sp_vertex_res);			
							break;
						}
					}
				}
			}
			////////////////////////////////////////////
			// FILL Cl OBJECT WITH CLUSTERED DATA

			Cluster Cl; // Object that will hold clustered and reclustered points

			if(is_selected){
				for(int i = 0 ; i < ord_trk.size() ; ++i){
					true_vertex = 0;
					Cluster_pt pt;
					pt.rn = run_num;
					pt.ev = ev_num;
					pt.cl = cluster;
					pt.x = ord_trk.at(i).x;
					pt.y = ord_trk.at(i).y;
					pt.z = ord_trk.at(i).z;
					pt.q = ord_trk.at(i).q;
					pt.uq = ord_trk.at(i).uq;
					pt.vq = ord_trk.at(i).vq;
					pt.wq = ord_trk.at(i).wq;
					pt.px = ord_trk.at(i).x;
					pt.py = ord_trk.at(i).y;
					pt.pz = ord_trk.at(i).z;
					pt.ord = 1.;
					pt.closeness = -1.;
					if(ord_trk.at(i).x == VertexPoint.x && ord_trk.at(i).y == VertexPoint.y && ord_trk.at(i).z == VertexPoint.z){
						pt.vertex = 1.;
					}else{
						pt.vertex = 0.;
					}
					if(ord_trk.at(i).x == sp_VertexPoint.x && ord_trk.at(i).y == sp_VertexPoint.y && ord_trk.at(i).z == sp_VertexPoint.z){
						pt.sp_vertex = 1.;
					}else{
						pt.sp_vertex = 0.;
					}
					for (int cand = 0; cand < mcand_size; ++cand){
						if(Michel_candidates[cand][0] == run_num && Michel_candidates[cand][1] == ev_num && Michel_candidates[cand][2] == cluster){
							if(Michel_candidates[cand][3] == 0 && Michel_candidates[cand][4] == 0 && Michel_candidates[cand][5] == 0) break;
							if(ord_trk.at(i).x > Michel_candidates[cand][3] + 0.001 || ord_trk.at(i).x < Michel_candidates[cand][3] - 0.001) continue;
							if(ord_trk.at(i).y > Michel_candidates[cand][4] + 0.001 || ord_trk.at(i).y < Michel_candidates[cand][4] - 0.001) continue;
							if(ord_trk.at(i).z > Michel_candidates[cand][5] + 0.001 || ord_trk.at(i).z < Michel_candidates[cand][5] - 0.001) continue;
							true_vertex = 1;
							true_vertex_found = true;
							true_ver_count += 1;
							break;
						}
					}
					pt.true_vertex = true_vertex;
					pt.is_ord = 1;
					Cl.push_back(pt);
				}
			}	

			//////////////////////////////
			// RECLUSTERING ALGORITHM
			//////////////////////////////

			if(is_selected){
				double dist_cluster, low_dist_cluster;
				double x1_diff, y1_diff, z1_diff;
				double x2_diff, y2_diff, z2_diff;
				double Rx,Ry,Rz,Rq;
				double Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,C,Vx,Vy,Vz;
				double normAB, normAV, normVP;
				double dotProd_AVAB;
				double Pq, Puq, Pvq, Pwq;
				int closest_line_seg; 
				bool closest_pt_to_line_found;
				vector<double> CrossP;
				double closeness;
				for(int i = 0; i < points_gd.size(); ++i){
					Cluster_pt temp_pt;
					low_dist_cluster = 100000.;
					closest_pt_to_line_found = false;
					Px = points_gd.at(i).x;
					Py = points_gd.at(i).y;
					Pz = points_gd.at(i).z;
					Pq = points_gd.at(i).q;
					Puq = points_gd.at(i).uq;
					Pvq = points_gd.at(i).vq;
					Pwq = points_gd.at(i).wq;
					for(int j = 0; j < Cl.size() - 1; ++j){
						Ax = Cl.at(j).px;
						Bx = Cl.at(j+1).px;
						Ay = Cl.at(j).py;
						By = Cl.at(j+1).py;
						Az = Cl.at(j).pz;
						Bz = Cl.at(j+1).pz;
											
						C = (Px-Ax)*(Bx-Ax) + (Py-Ay)*(By-Ay) + (Pz-Az)*(Bz-Az);
						C = C/(pow((Bx-Ax),2.0) + pow((By-Ay),2.0) + pow((Bz-Az),2.0));
						
						Vx = Ax + C*(Bx-Ax);
						Vy = Ay + C*(By-Ay);
						Vz = Az + C*(Bz-Az);
	
						normAB = Pythagoras((Bx-Ax),0.0,(By-Ay),0.0,(Bz-Az),0.0);
						normAV = Pythagoras((Vx-Ax),0.0,(Vy-Ay),0.0,(Vz-Az),0.0);
					
						dotProd_AVAB = (Vx-Ax)*(Bx-Ax) + (Vy-Ay)*(By-Ay) + (Vz-Az)*(Bz-Az);
						dotProd_AVAB = dotProd_AVAB/(normAB*normAV);	
						normVP = Pythagoras(Px,Vx,Py,Vy,Pz,Vz);
						
						if(dotProd_AVAB == -1) continue;
						if(normAV > normAB) continue;
						if(max_dist_for_charge < normVP) continue;	

						if(normVP < low_dist_cluster){
							low_dist_cluster = normVP;
							closest_line_seg = j;
							closest_pt_to_line_found = true;
						}
					}

					if(closest_pt_to_line_found == false) continue;

					Ax = Cl.at(closest_line_seg).px;
					Bx = Cl.at(closest_line_seg+1).px;
					Ay = Cl.at(closest_line_seg).py;
					By = Cl.at(closest_line_seg+1).py;
					Az = Cl.at(closest_line_seg).pz;
					Bz = Cl.at(closest_line_seg+1).pz;
					C = (Px-Ax)*(Bx-Ax) + (Py-Ay)*(By-Ay) + (Pz-Az)*(Bz-Az);
					C = C/(pow((Bx-Ax),2.0) + pow((By-Ay),2.0) + pow((Bz-Az),2.0));
					Vx = Ax + C*(Bx-Ax);
					Vy = Ay + C*(By-Ay);
					Vz = Az + C*(Bz-Az);
					
					//Store reclustered data
					temp_pt.rn = run_num;
					temp_pt.ev = ev_num;
					temp_pt.cl = cluster;
					temp_pt.x = Px;
					temp_pt.y = Py;
					temp_pt.z = Pz;
					temp_pt.q = Pq;

					temp_pt.uq = Puq;
					temp_pt.vq = Pvq;
					temp_pt.wq = Pwq;

					temp_pt.px = Vx;
					temp_pt.py = Vy;
					temp_pt.pz = Vz;
					temp_pt.ord = 0.;
					temp_pt.closeness = low_dist_cluster;
					temp_pt.vertex = 0.;
					temp_pt.sp_vertex = 0.;
					true_vertex = 0;
					if(true_vertex_found == false){
						for (int cand = 0; cand < mcand_size; ++cand){
							if(Michel_candidates[cand][0] == run_num && Michel_candidates[cand][1] == ev_num && Michel_candidates[cand][2] == cluster){
								if(Michel_candidates[cand][3] == 0 && Michel_candidates[cand][4] == 0 && Michel_candidates[cand][5] == 0) break;
								if(Px > Michel_candidates[cand][3] + 0.001 || Px < Michel_candidates[cand][3] - 0.001) continue;
								if(Py > Michel_candidates[cand][4] + 0.001 || Py < Michel_candidates[cand][4] - 0.001) continue;
								if(Pz > Michel_candidates[cand][5] + 0.001 || Pz < Michel_candidates[cand][5] - 0.001) continue;
								true_vertex = 1;
								true_ver_count += 1;
								break;
							}
						}
					}
					temp_pt.true_vertex = true_vertex;
					temp_pt.is_ord = 0.;
					Cl.insert(Cl.begin() + closest_line_seg+1, temp_pt);					
				}

				/////////////////////////////////
				// Physics calculations
				is_el = false;
				el_energy = 0;
				el_energy_u = 0;
				el_energy_v = 0;
				el_energy_w = 0;
				mu_energy = 0;
				t_dQ = 0.;
				t_dE = 0.;
				residual_range = 0.;
				track_range = 0.;
				dQ = 0.;
				duQ = 0.;
				dvQ = 0.;
				dwQ = 0.;
				dE = 0.;
				duE = 0.;
				dvE = 0.;
				dwE = 0.;
				dx = 0.;
				double per_25;
				
				for(int i = 0; i < Cl.size(); ++i){
					if(i > 0) track_range += Pythagoras(Cl.at(i).px,Cl.at(i-1).px,Cl.at(i).py,Cl.at(i-1).py,Cl.at(i).pz,Cl.at(i-1).pz);
				}

				for(int i = 0; i < Cl.size(); ++i){
					//TNtuple *nt_plane_q = 	new TNtuple("nt_plane_q","nt_plane_q","run_num:ev_num:cluster_id:x:y:z:q:uq:vq:wq:px:py:pz");
					nt_plane_q -> Fill(run_num,ev_num,cluster,Cl.at(i).x,Cl.at(i).y,Cl.at(i).z,Cl.at(i).q,Cl.at(i).uq,Cl.at(i).vq,Cl.at(i).wq,Cl.at(i).px,Cl.at(i).py,Cl.at(i).pz);
					nt_clus_trk -> Fill(run_num,ev_num,cluster,Cl.at(i).x,Cl.at(i).y,Cl.at(i).z,Cl.at(i).q,Cl.at(i).px,Cl.at(i).py,Cl.at(i).pz,Cl.at(i).ord,Cl.at(i).closeness,Cl.at(i).vertex,Cl.at(i).sp_vertex,Cl.at(i).true_vertex);
					if(i > 0) dx = Pythagoras(Cl.at(i).px,Cl.at(i-1).px,Cl.at(i).py,Cl.at(i-1).py,Cl.at(i).pz,Cl.at(i-1).pz);
					if(i > 0) track_range -= dx;
					if(i > q_win && i < Cl.size() - q_win){
						for(int j = i - q_win - 1; j < i + q_win; ++j){
							v_q.push_back(Cl.at(j).q);
							v_uq.push_back(Cl.at(j).uq);
							v_vq.push_back(Cl.at(j).vq);
							v_wq.push_back(Cl.at(j).wq);
						}
						per_25 = v_q.size()*0.25;
						per_25 = (int)per_25;

						std::sort(v_q.begin(), v_q.end());
						std::sort(v_uq.begin(), v_uq.end());
						std::sort(v_vq.begin(), v_vq.end());
						std::sort(v_wq.begin(), v_wq.end());


						for(int k = per_25; k < v_q.size() - per_25; ++k){
							dQ  += v_q[k];
							duQ += v_uq[k];
							dvQ += v_vq[k];
							dwQ += v_wq[k];
						}
						dQ = dQ/(double)(v_q.size()-2*per_25);
						duQ = duQ/(double)(v_uq.size()-2*per_25);
						dvQ = dvQ/(double)(v_vq.size()-2*per_25);
						dwQ = dwQ/(double)(v_wq.size()-2*per_25);
						dE = dQ * cal_factor; 
						duE = duQ * cal_factor; 
						dvE = dvQ * cal_factor; 
						dwE = dwQ * cal_factor; 
						dQdx = dQ/dx;
						duQdx = duQ/dx;
						dvQdx = dvQ/dx;
						dwQdx = dwQ/dx;
						dEdx = dE/dx;
						duEdx = duE/dx;
						dvEdx = dvE/dx;
						dwEdx = dwE/dx;
						nt_ddx_cl -> Fill(run_num,ev_num,cluster,track_range,dQ,dx,dQdx,dEdx);
						//TNtuple *nt_plane_dQdx_cl = new TNtuple("nt_plane_dQdx_cl","nt_plane_dQdx_cl","run_num:ev_num:cluster_id:rr:duQ:dvQ:dwQ:dx:duQdx:dvQdx:dwQdx");
						nt_plane_dQdx_cl -> Fill(run_num,ev_num,cluster,track_range,dQ,duQ,dvQ,dwQ,dx,duQdx,dvQdx,dwQdx);
						nt_plane_dEdx_cl -> Fill(run_num,ev_num,cluster,track_range,dE,duE,dvE,dwE,dx,duEdx,dvEdx,dwEdx);
						dQ = 0.;
						duQ = 0.;
						dvQ = 0.;
						dwQ = 0.;
						dE = 0.;
						duE = 0.;
						dvE = 0.;
						dwE = 0.;
						dx = 0.;
						v_q.clear();
						v_uq.clear();
						v_vq.clear();
						v_wq.clear();
					}
					if(Cl.at(i).sp_vertex == 1.) is_el = true;
					if(is_el){
						el_energy += Cl.at(i).q * cal_factor;
						el_energy_u += Cl.at(i).uq * cal_factor;
						el_energy_v += Cl.at(i).vq * cal_factor;
						el_energy_w += Cl.at(i).wq * cal_factor;
					}else{
						mu_energy += Cl.at(i).q * cal_factor;
					}	
				}

				nt_cluster_info -> Fill(run_num,ev_num,cluster,el_energy,el_energy_u,el_energy_v,el_energy_w,mu_energy,vertex_res,sp_vertex_res);	
			}		    	
		}
		infile->Close();
	}


	//////////////////////////////
	// Study Numbers
	//////////////////////////////

	purity = ((float)michel_count)/((float)track_selected_as_michel);
	efficiency = ((float)michel_count)/((float)(s-1));
	ver_rms = sqrt((1./((float)michel_count)) * ver_rms);
	ver_mean = (1./((float)michel_count)) * ver_mean;
	//Standard Deviation Calculation
	for (int i = 0; i < v_vertex_res.size(); ++i){
		ver_sd += pow(v_vertex_res[i] - ver_mean, 2.0);
	}
	ver_sd = sqrt(ver_sd/((float)v_vertex_res.size()));

	sp_ver_rms = sqrt((1./((float)michel_count)) * sp_ver_rms);
	sp_ver_mean = (1./((float)michel_count)) * sp_ver_mean;
	//Standard Deviation Calculation SP
	for (int i = 0; i < v_sp_vertex_res.size(); ++i){
		sp_ver_sd += pow(v_sp_vertex_res[i] - sp_ver_mean, 2.0);
	}
	sp_ver_sd = sqrt(sp_ver_sd/((float)v_sp_vertex_res.size()));

	if(isnan(purity)) purity = 0.;
	if(isnan(ver_rms)) ver_rms = 0.;
	if(isnan(ver_mean)) ver_mean = 0.;


	cout << "###################################################################" << endl;
	cout << "Total number of tracks = " << total_num_tracks << "; Number of Michels in sample = " << s - 1 << endl;
	cout << "Tracks smaller than the pca window = " << small_tracks << endl;
	cout << "###################################################################" << endl;
	cout << "################### Ordering of Tracks ############################" << endl;
	cout << "###################################################################" << endl;
	cout << "Tracks after ordering algorithm = " << tracks_survived_ord_alg << " with alpha = " << alpha << " and ext_alpha = " << ext_alpha << endl;
	cout << "Michel clusters that survived the ordering algorithm = " << michels_survived_ord_alg << endl;
	cout << "Michel clusters that were cut out by ordering algorithm = " << ord_alg_cutouts_michels << endl;
	cout << "Cone Angle Criterion: Cos(/phi) > " << min_cone_ang << " => /phi < " << max_phi << endl;
	cout << "Unclustered Porcentage Cut > " << cut_ucp_ratio << endl;
	cout << "Points considered for PCA calculations: " << ang_points << endl;
	cout << "Epsilon phi = "<< eps_ang <<"; Epsilon dist = " << eps_dist << endl;
	cout << "###################################################################" << endl;
	cout << "###################### Michel ID part #############################" << endl;
	cout << "###################################################################" << endl;
	cout << "Tracks that were selected as Michels  = " << track_selected_as_michel << endl;
	cout << "Tracks correctly selected as selected as Michels = " << michel_count << endl;
	cout << "# of points used in prev and post windows: " << prev_win_size << ", " << post_win_size << endl;
	cout << "Cuts applied on Cos(/theta): Min = " << min_costheta << ", Max = " << max_costheta << endl;
	cout << "Cuts applied on /theta: Max = " << acos(min_costheta) * 180./ PI << ", Min = " << acos(max_costheta) * 180./ PI << endl;
	cout << "Vertex RMS = " << ver_rms << ", Vertex Mean = " << ver_mean << ", Vertex sigma = " << ver_sd << endl;
	cout << "###################################################################" << endl;
	cout << "###################### SP Vertex Calculation ######################" << endl;
	cout << "###################################################################" << endl;
	cout << "SP Vertex RMS = " << sp_ver_rms << ", SP Vertex Mean = " << sp_ver_mean << ", SP Vertex sigma = " << sp_ver_sd << endl;
	cout << "###################################################################" << endl;
	cout << "########################## Results ################################" << endl;
	cout << "###################################################################" << endl;
	cout << "Purity = " << purity << ", Efficiency = " << efficiency << endl;
	cout << "####################################################################" << endl;
	cout << endl;

	nt_results->Fill(tracks_survived_ord_alg,alpha,michels_survived_ord_alg,min_cone_ang,max_phi,ang_points,track_selected_as_michel,michel_count,min_costheta,theta,purity,efficiency,ver_rms,ver_mean);	
	f_output->Write();
	f_output->Close();
	cout << "DONE" << endl;
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////END OF MAIN PROGRAM/////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////START OF FUNCTIONS//////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

double Pythagoras(double x1,double x2,double y1,double y2,double z1,double z2){
	double dist;
	dist = sqrt(pow(x2-x1,2.) + pow(y2-y1,2.) + pow(z2-z1,2.));
	return dist;
}

vector<double> Unit_Vec(double x1,double y1,double z1){
	std::vector<double> v;
	double norm;
	norm = Pythagoras(x1,0.0,y1,0.0,z1,0.0);
	v.push_back(x1/norm);
	v.push_back(y1/norm);
	v.push_back(z1/norm);
	return v;
}

vector<double> Unit_Vec_NO(double x1,double x2,double y1,double y2,double z1,double z2){
	std::vector<double> v;
	double norm;
	norm = Pythagoras(x1,x2,y1,y2,z1,z2);
	v.push_back((x2-x1)/norm);
	v.push_back((y2-y1)/norm);
	v.push_back((z2-z1)/norm);
	return v;
}

double dotProdFunc(double x1,double x2,double y1,double y2,double z1,double z2){
	double dotP;
	dotP = x1*x2 + y1*y2 + z1*z2;
	return dotP;
}

vector<double> CrossProd(double x1,double x2,double y1,double y2,double z1,double z2){
	vector<double> v;
	double x,y,z;
	x = y1*z2 - z1*y2;
	y = x1*z2 - x2*z1;
	y = -y;
	z = x1*y2 - x2*y1;	
	v.push_back(x);
	v.push_back(y);
	v.push_back(z);	
	return v;
}

void LoadPointCloud(PointCloud &points, const track_def &ord_trk) {
	for (int i = 0; i < ord_trk.size(); ++i){
		Point tempPoint;
		tempPoint.x = ord_trk.at(i).x;
		tempPoint.y = ord_trk.at(i).y;
		tempPoint.z = ord_trk.at(i).z;
		tempPoint.q = ord_trk.at(i).q;
		points.push_back(tempPoint);

	}
	return;
}
	
PCAResults DoPCA(const PointCloud &points) {
	TVector3 outputCentroid;
	pair<TVector3,TVector3> outputEndPoints;
	float outputLength;
	TVector3 outputEigenValues;
	vector<TVector3> outputEigenVecs;
	float meanPosition[3] = {0., 0., 0.};
	unsigned int nThreeDHits = 0;
	for (unsigned int i = 0; i < points.size(); i++) {
		meanPosition[0] += points[i].x;
		meanPosition[1] += points[i].y;
		meanPosition[2] += points[i].z;
		++nThreeDHits;
	}
	if (nThreeDHits == 0) {
		PCAResults results;
		return results; 
	}
	const float nThreeDHitsAsFloat(static_cast<float>(nThreeDHits));
	meanPosition[0] /= nThreeDHitsAsFloat;
	meanPosition[1] /= nThreeDHitsAsFloat;
	meanPosition[2] /= nThreeDHitsAsFloat;
	outputCentroid = TVector3(meanPosition[0], meanPosition[1], meanPosition[2]);
	float xi2 = 0.0;
	float xiyi = 0.0;
	float xizi = 0.0;
	float yi2 = 0.0;
	float yizi = 0.0;
	float zi2 = 0.0;
	float weightSum = 0.0;
	for (unsigned int i = 0; i < points.size(); i++) {
		const float weight(1.);
		const float x((points[i].x - meanPosition[0]) * weight);
		const float y((points[i].y - meanPosition[1]) * weight);
		const float z((points[i].z - meanPosition[2]) * weight);
		xi2	+= x * x;
		xiyi += x * y;
		xizi += x * z;
		yi2	+= y * y;
		yizi += y * z;
		zi2	+= z * z;
		weightSum += weight * weight;
	}

	Eigen::Matrix3f sig;

	sig <<  xi2, xiyi, xizi,
			xiyi, yi2, yizi,
			xizi, yizi, zi2;

	sig *= 1.0 / weightSum;

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenMat(sig);

	typedef std::pair<float,size_t> EigenValColPair;
	typedef std::vector<EigenValColPair> EigenValColVector;

	EigenValColVector eigenValColVector;
	const auto &resultEigenMat(eigenMat.eigenvalues());
	eigenValColVector.emplace_back(resultEigenMat(0), 0);
	eigenValColVector.emplace_back(resultEigenMat(1), 1);
	eigenValColVector.emplace_back(resultEigenMat(2), 2);

	std::sort(eigenValColVector.begin(), eigenValColVector.end(), [](const EigenValColPair &left, const EigenValColPair &right){return left.first > right.first;} );

	outputEigenValues = TVector3(eigenValColVector.at(0).first, eigenValColVector.at(1).first, eigenValColVector.at(2).first);

	const Eigen::Matrix3f &eigenVecs(eigenMat.eigenvectors());

	for (const EigenValColPair &pair : eigenValColVector) {
		outputEigenVecs.emplace_back(eigenVecs(0, pair.second), eigenVecs(1, pair.second), eigenVecs(2, pair.second));
	}

	PCAResults results;

	Eigen::ParametrizedLine<float,3> priAxis(Eigen::Vector3f(outputCentroid(0),outputCentroid(1),outputCentroid(2)),Eigen::Vector3f(outputEigenVecs[0](0),outputEigenVecs[0](1),outputEigenVecs[0](2)));

	Eigen::Vector3f endPoint1(Eigen::Vector3f(outputCentroid(0),outputCentroid(1),outputCentroid(2)));
	Eigen::Vector3f endPoint2(Eigen::Vector3f(outputCentroid(0),outputCentroid(1),outputCentroid(2)));

	Eigen::Vector3f testPoint;
	Eigen::Vector3f projTestPoint;
	float maxDist1 = -1.0;
	float maxDist2 = -1.0;
	float dist;
	float dotP;
	for (unsigned int i = 0; i < points.size(); i++) {
		testPoint = Eigen::Vector3f(points[i].x,points[i].y,points[i].z);
		projTestPoint = priAxis.projection(testPoint);
		dist = sqrt(pow(projTestPoint(0)-outputCentroid(0),2.0)+pow(projTestPoint(1)-outputCentroid(1),2.0)+pow(projTestPoint(2)-outputCentroid(2),2.0));
		dotP = (projTestPoint(0)-outputCentroid(0))*outputEigenVecs[0](0) + (projTestPoint(1)-outputCentroid(1))*outputEigenVecs[0](1) + (projTestPoint(2)-outputCentroid(2))*outputEigenVecs[0](2);


		if ((dotP < 0.0) && (dist > maxDist1)) {
			endPoint1 = projTestPoint;
			maxDist1 = dist;
		}
		else if ((dotP > 0.0) && (dist > maxDist2)) {
			endPoint2 = projTestPoint;
			maxDist2 = dist;
		}
	}
	outputEndPoints.first = TVector3(endPoint1(0),endPoint1(1),endPoint1(2));
	outputEndPoints.second = TVector3(endPoint2(0),endPoint2(1),endPoint2(2));
	outputLength = sqrt(pow(endPoint2(0)-endPoint1(0),2.0)+pow(endPoint2(1)-endPoint1(1),2.0)+pow(endPoint2(2)-endPoint1(2),2.0));
	results.centroid = outputCentroid;
	results.endPoints = outputEndPoints;
	results.length = outputLength;
	results.eVals = outputEigenValues;
	results.eVecs = outputEigenVecs;
	return results;
}
