/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_H_
#define PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_H_

#include <pcl/pcl_macros.h>
#include <pcl/filters/boost.h>
#include <pcl/filters/voxel_grid.h>
#include <map>
#include <unordered_map>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace pclomp
{
  /** \brief A searchable voxel structure containing the mean and covariance of the data.
    * \note For more information please see
    * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
    * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
    * PhD thesis, Orebro University. Orebro Studies in Technology 36</b>
    * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
    */
  template<typename PointT>
  class IncrementalVoxelGridCovariance : public pcl::VoxelGrid<PointT>
  {
    protected:
      using pcl::VoxelGrid<PointT>::filter_name_;
      using pcl::VoxelGrid<PointT>::getClassName;
      using pcl::VoxelGrid<PointT>::input_;
      using pcl::VoxelGrid<PointT>::indices_;
      using pcl::VoxelGrid<PointT>::filter_limit_negative_;
      using pcl::VoxelGrid<PointT>::filter_limit_min_;
      using pcl::VoxelGrid<PointT>::filter_limit_max_;
      using pcl::VoxelGrid<PointT>::filter_field_name_;

      using pcl::VoxelGrid<PointT>::downsample_all_data_;
      using pcl::VoxelGrid<PointT>::leaf_layout_;
      using pcl::VoxelGrid<PointT>::save_leaf_layout_;
      using pcl::VoxelGrid<PointT>::leaf_size_;
      using pcl::VoxelGrid<PointT>::min_b_;
      using pcl::VoxelGrid<PointT>::max_b_;
      using pcl::VoxelGrid<PointT>::inverse_leaf_size_;
      using pcl::VoxelGrid<PointT>::div_b_;
      using pcl::VoxelGrid<PointT>::divb_mul_;

      typedef typename pcl::traits::fieldList<PointT>::type FieldList;
      typedef typename pcl::Filter<PointT>::PointCloud PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    public:

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
      typedef pcl::shared_ptr< pcl::VoxelGrid<PointT> > Ptr;
      typedef pcl::shared_ptr< const pcl::VoxelGrid<PointT> > ConstPtr;
#else
      typedef boost::shared_ptr< pcl::VoxelGrid<PointT> > Ptr;
      typedef boost::shared_ptr< const pcl::VoxelGrid<PointT> > ConstPtr;
#endif

      /** \brief Simple structure to hold a centroid, covariance and the number of points in a leaf .
        * Inverse covariance, eigen vectors and eigen values are precomputed. */
      struct Leaf
      {
        /** \brief Constructor.
         * Sets \ref nr_points, \ref icov_, \ref mean_ and \ref evals_ to 0 and \ref cov_ and \ref evecs_ to the identity matrix
         */
        Leaf () :
          nr_points (0),
          centroid (),
          mean_ (Eigen::Vector3d::Zero ()),
          cov_ (Eigen::Matrix3d::Identity ()),
          icov_ (Eigen::Matrix3d::Zero ()),
          evecs_ (Eigen::Matrix3d::Identity ()),
          evals_ (Eigen::Vector3d::Zero ()),
          inited_ (false),
          centroid_sum_ (),
          pt_sum_ (Eigen::Vector3d::Zero ()),
          pt_sq_sum_ (Eigen::Matrix3d::Zero ())
        {
        }

        /** \brief Get the voxel covariance.
          * \return covariance matrix
          */
        Eigen::Matrix3d
        getCov () const
        {
          return (cov_);
        }

        /** \brief Get the inverse of the voxel covariance.
          * \return inverse covariance matrix
          */
        Eigen::Matrix3d
        getInverseCov () const
        {
          return (icov_);
        }

        /** \brief Get the voxel centroid.
          * \return centroid
          */
        Eigen::Vector3d
        getMean () const
        {
          return (mean_);
        }

        /** \brief Get the eigen vectors of the voxel covariance.
          * \note Order corresponds with \ref getEvals
          * \return matrix whose columns contain eigen vectors
          */
        Eigen::Matrix3d
        getEvecs () const
        {
          return (evecs_);
        }

        /** \brief Get the eigen values of the voxel covariance.
          * \note Order corresponds with \ref getEvecs
          * \return vector of eigen values
          */
        Eigen::Vector3d
        getEvals () const
        {
          return (evals_);
        }

        /** \brief Get the number of points contained by this voxel.
          * \return number of points
          */
        int
        getPointCount () const
        {
          return (nr_points);
        }

        /** \brief Number of points contained by voxel */
        int nr_points;

        /** \brief Nd voxel centroid
         * \note Differs from \ref mean_ when color data is used
         */
        Eigen::VectorXf centroid;

        /** \brief 3D voxel centroid */
        Eigen::Vector3d mean_;

        /** \brief Voxel covariance matrix */
        Eigen::Matrix3d cov_;

        /** \brief Inverse of voxel covariance matrix */
        Eigen::Matrix3d icov_;

        /** \brief Eigen vectors of voxel covariance matrix */
        Eigen::Matrix3d evecs_;

        /** \brief Eigen values of voxel covariance matrix */
        Eigen::Vector3d evals_;


        /** **********************************************************************
         * [WANG_Guanhua] variables below are added to support incremental update!
         * Basically, below variables are used to save 'last state'.
        */

        /** \brief [WANG_Guanhua] true if already got mean & cov, in this case 
         * incremental update is required.
         */
        bool inited_;

        /** \brief [WANG_Guanhua] Lifelong sum of "centroid". */
        Eigen::VectorXf centroid_sum_;

        /** \brief [WANG_Guanhua] Lifelong sum of "point". */
        Eigen::Vector3d pt_sum_;

        /** \brief [WANG_Guanhua] Lifelong sum of "point square matrix". */
        Eigen::Matrix3d pt_sq_sum_;

        /** \brief [WANG_Guanhua] Update mean & cov, compatiable with both usual mode and incremental mode. */
        void UpdateState(const int& nun_points_required, const double& min_eigenvalue_ratio = 0.01) {
          // Eigen values and vectors calculated to prevent near singular matrices.
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
          Eigen::Matrix3d eigen_val;

          double min_covar_eigvalue;

          // compute gaussian params as usual.
          centroid = centroid_sum_ / static_cast<float> (nr_points);
          mean_ = pt_sum_ / nr_points;

          if (nr_points >= nun_points_required) {
            // Single pass covariance calculation
            cov_ = (pt_sq_sum_ - 2 * (pt_sum_ * mean_.transpose ())) / nr_points + mean_ * mean_.transpose ();
            cov_ *= (nr_points - 1.0) / nr_points;

            //Normalize Eigen Val such that max no more than 100x min.
            eigensolver.compute (cov_);
            eigen_val = eigensolver.eigenvalues ().asDiagonal ();
            evecs_ = eigensolver.eigenvectors ();

            // unexpected situation.
            if (eigen_val (0, 0) < 0 || eigen_val (1, 1) < 0 || eigen_val (2, 2) <= 0) {
              nr_points = -1;
              return;
            }

            // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
            min_covar_eigvalue = min_eigenvalue_ratio * eigen_val (2, 2);
            if (eigen_val (0, 0) < min_covar_eigvalue) {
              eigen_val (0, 0) = min_covar_eigvalue;

              if (eigen_val (1, 1) < min_covar_eigvalue) {
                eigen_val (1, 1) = min_covar_eigvalue;
              }
              // reset covariance matrix from smoothed eigen values.
              cov_ = evecs_ * eigen_val * evecs_.inverse ();
            }
            evals_ = eigen_val.diagonal ();
            icov_ = cov_.inverse ();
            // unexpected situation.
            if (icov_.maxCoeff () == std::numeric_limits<float>::infinity ( )
                || icov_.minCoeff () == -std::numeric_limits<float>::infinity ( ) )
            {
              nr_points = -1;
              return;
            }
          }
        }

      }; // struct Leaf

      /** \brief Pointer to IncrementalVoxelGridCovariance leaf structure */
      typedef Leaf* LeafPtr;

      /** \brief Const pointer to IncrementalVoxelGridCovariance leaf structure */
      typedef const Leaf* LeafConstPtr;

      typedef std::map<size_t, Leaf> Map;

    public:

      /** \brief Constructor.
       * Sets \ref leaf_size_ to 0 and \ref searchable_ to false.
       */
      IncrementalVoxelGridCovariance () :
        searchable_ (false),
        min_points_per_voxel_ (6),
        min_covar_eigvalue_mult_ (0.01),
        leaves_ (),
        voxel_centroids_ (),
        voxel_centroids_leaf_indices_ (),
        kdtree_ (),
        enable_incremetal_mode_ (true),
        enable_voxel_downsample_ (false),
        voxel_downsample_size_ (0.1),
        bounding_box_size_ (Eigen::Vector3f(200.f, 200.f, 100.f)), 
        trim_every_n_meters_ (10.f),
        current_position_ (Eigen::Vector3f(0.f, 0.f, 0.f)),
        last_trimmed_position_ (Eigen::Vector3f(0.f, 0.f, 0.f)),
        generate_voxel_centroid_cloud_ (false),
        enable_obb_update_ (false),
        print_verbose_info_ (false),
        verbose_info_level_ (-1)
      {
        downsample_all_data_ = false;
        save_leaf_layout_ = false;
        leaf_size_.setZero ();
        min_b_.setZero ();
        max_b_.setZero ();
        filter_name_ = "IncrementalVoxelGridCovariance";
      }

      /** \brief Set the minimum number of points required for a cell to be used (must be 3 or greater for covariance calculation).
        * \param[in] min_points_per_voxel the minimum number of points for required for a voxel to be used
        */
      inline void
      setMinPointPerVoxel (int min_points_per_voxel)
      {
        if(min_points_per_voxel > 2)
        {
          min_points_per_voxel_ = min_points_per_voxel;
        }
        else
        {
          PCL_WARN ("%s: Covariance calculation requires at least 3 points, setting Min Point per Voxel to 3 ", this->getClassName ().c_str ());
          min_points_per_voxel_ = 3;
        }
      }

      /** \brief Get the minimum number of points required for a cell to be used.
        * \return the minimum number of points for required for a voxel to be used
        */
      inline int
      getMinPointPerVoxel ()
      {
        return min_points_per_voxel_;
      }

      /** \brief Set the minimum allowable ratio between eigenvalues to prevent singular covariance matrices.
        * \param[in] min_covar_eigvalue_mult the minimum allowable ratio between eigenvalues
        */
      inline void
      setCovEigValueInflationRatio (double min_covar_eigvalue_mult)
      {
        min_covar_eigvalue_mult_ = min_covar_eigvalue_mult;
      }

      /** \brief Get the minimum allowable ratio between eigenvalues to prevent singular covariance matrices.
        * \return the minimum allowable ratio between eigenvalues
        */
      inline double
      getCovEigValueInflationRatio ()
      {
        return min_covar_eigvalue_mult_;
      }

      /** \brief Filter cloud and initializes voxel structure.
       * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
       * \param[in] searchable flag if voxel structure is searchable, if true then kdtree is built
       */
      inline void
      filter (PointCloud &output, bool searchable = false)
      {
        generate_voxel_centroid_cloud_ = true;
        searchable_ = searchable;
        applyFilter (output);

        voxel_centroids_ = PointCloudPtr (new PointCloud (output));

        if (searchable_ && voxel_centroids_->size() > 0)
        {
          // Initiates kdtree of the centroids of voxels containing a sufficient number of points
          // [WANG_Guanhua] Note that this may take some time and affect computation efficiency.
          kdtree_.setInputCloud (voxel_centroids_);
        }
      }

      /** \brief Initializes voxel structure.
       * \param[in] searchable flag if voxel structure is searchable, if true then kdtree is built
       */
      inline void
      filter (bool searchable = false)
      {
        generate_voxel_centroid_cloud_ = searchable;
        searchable_ = searchable;
        voxel_centroids_ = PointCloudPtr (new PointCloud);
        applyFilter (*voxel_centroids_);

        if (searchable_ && voxel_centroids_->size() > 0)
        {
          // Initiates kdtree of the centroids of voxels containing a sufficient number of points
          // [WANG_Guanhua] Note that this may take some time and affect computation efficiency.
          kdtree_.setInputCloud (voxel_centroids_);
        }
      }

      /** \brief Get the voxel containing point p.
       * \param[in] index the index of the leaf structure node
       * \return const pointer to leaf structure
       */
      inline LeafConstPtr
      getLeaf (int index)
      {
        auto leaf_iter = leaves_.find (index);
        if (leaf_iter != leaves_.end ())
        {
          LeafConstPtr ret (&(leaf_iter->second));
          return ret;
        }
        else
          return NULL;
      }

      /** \brief Get the voxel containing point p.
       * \param[in] p the point to get the leaf structure at
       * \return const pointer to leaf structure
       */
      inline LeafConstPtr
      getLeaf (PointT &p)
      {
        // Generate index associated with p
        int ijk0 = static_cast<int> (floor (p.x * inverse_leaf_size_[0]) - min_b_[0]);
        int ijk1 = static_cast<int> (floor (p.y * inverse_leaf_size_[1]) - min_b_[1]);
        int ijk2 = static_cast<int> (floor (p.z * inverse_leaf_size_[2]) - min_b_[2]);

        // Compute the centroid leaf index
        int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

        // Find leaf associated with index
        auto leaf_iter = leaves_.find (idx);
        if (leaf_iter != leaves_.end ())
        {
          // If such a leaf exists return the pointer to the leaf structure
          LeafConstPtr ret (&(leaf_iter->second));
          return ret;
        }
        else
          return NULL;
      }

      /** \brief Get the voxel containing point p.
       * \param[in] p the point to get the leaf structure at
       * \return const pointer to leaf structure
       */
      inline LeafConstPtr
      getLeaf (Eigen::Vector3f &p)
      {
        // Generate index associated with p
        int ijk0 = static_cast<int> (floor (p[0] * inverse_leaf_size_[0]) - min_b_[0]);
        int ijk1 = static_cast<int> (floor (p[1] * inverse_leaf_size_[1]) - min_b_[1]);
        int ijk2 = static_cast<int> (floor (p[2] * inverse_leaf_size_[2]) - min_b_[2]);

        // Compute the centroid leaf index
        int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

        // Find leaf associated with index
        auto leaf_iter = leaves_.find (idx);
        if (leaf_iter != leaves_.end ())
        {
          // If such a leaf exists return the pointer to the leaf structure
          LeafConstPtr ret (&(leaf_iter->second));
          return ret;
        }
        else
          return NULL;

      }

      /** \brief Get the voxels surrounding point p, not including the voxel containing point p.
       * \note Only voxels containing a sufficient number of points are used (slower than radius search in practice).
       * \param[in] reference_point the point to get the leaf structure at
       * \param[out] neighbors
       * \return number of neighbors found
       */
      int getNeighborhoodAtPoint(const Eigen::MatrixXi&, const PointT& reference_point, std::vector<LeafConstPtr> &neighbors) const ;
      int getNeighborhoodAtPoint(const PointT& reference_point, std::vector<LeafConstPtr> &neighbors) const ;
      int getNeighborhoodAtPoint7(const PointT& reference_point, std::vector<LeafConstPtr> &neighbors) const ;
      int getNeighborhoodAtPoint1(const PointT& reference_point, std::vector<LeafConstPtr> &neighbors) const ;

      /** \brief Get the leaf structure map
       * \return a map containing all leaves
       */
      inline const Map&
      getLeaves ()
      {
        return leaves_;
      }

      /** \brief Get a pointcloud containing the voxel centroids
       * \note Only voxels containing a sufficient number of points are used.
       * \return a map containing all leaves
       */
      inline PointCloudPtr
      getCentroids ()
      {
        return voxel_centroids_;
      }

      /** \brief Get a cloud to visualize each voxels normal distribution.
       * \param[out] cell_cloud a cloud created by sampling the normal distributions of each voxel
       */
      void
      getDisplayCloud (pcl::PointCloud<pcl::PointXYZI>& cell_cloud, const int& num_points_per_voxel = 100);

      /// TODO: @wgh 重写下边的4个函数（实际是2个），以适应incremental版本。

      /** \brief Search for the k-nearest occupied voxels for the given query point.
       * \note Only voxels containing a sufficient number of points are used.
       * \param[in] point the given query point
       * \param[in] k the number of neighbors to search for
       * \param[out] k_leaves the resultant leaves of the neighboring points
       * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
       * \return number of neighbors found
       */
      int
      nearestKSearch (const PointT &point, int k,
                      std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances)
      {
        k_leaves.clear ();

        if (enable_incremetal_mode_) {
          PCL_WARN ("%s: Using kdtree in incremental mode, may affect computation efficiency.", 
            this->getClassName ().c_str ());
          return 0; /// TODO @wgh
        }

        // Check if kdtree has been built
        if (!searchable_)
        {
          PCL_WARN ("%s: Not Searchable", this->getClassName ().c_str ());
          return 0;
        }

        // Find k-nearest neighbors in the occupied voxel centroid cloud
        std::vector<int> k_indices;
        k = kdtree_.nearestKSearch (point, k, k_indices, k_sqr_distances);

        // Find leaves corresponding to neighbors
        k_leaves.reserve (k);
        for (std::vector<int>::iterator iter = k_indices.begin (); iter != k_indices.end (); iter++)
        {
          k_leaves.push_back (&leaves_[voxel_centroids_leaf_indices_[*iter]]);
        }
        return k;
      }

      /** \brief Search for the k-nearest occupied voxels for the given query point.
       * \note Only voxels containing a sufficient number of points are used.
       * \param[in] cloud the given query point
       * \param[in] index the index
       * \param[in] k the number of neighbors to search for
       * \param[out] k_leaves the resultant leaves of the neighboring points
       * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
       * \return number of neighbors found
       */
      inline int
      nearestKSearch (const PointCloud &cloud, int index, int k,
                      std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances)
      {
        if (index >= static_cast<int> (cloud.points.size ()) || index < 0)
          return (0);
        return (nearestKSearch (cloud.points[index], k, k_leaves, k_sqr_distances));
      }


      /** \brief Search for all the nearest occupied voxels of the query point in a given radius.
       * \note Only voxels containing a sufficient number of points are used.
       * \param[in] point the given query point
       * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
       * \param[out] k_leaves the resultant leaves of the neighboring points
       * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
       * \param[in] max_nn
       * \return number of neighbors found
       */
      int
      radiusSearch (const PointT &point, double radius, std::vector<LeafConstPtr> &k_leaves,
                    std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
      {
        k_leaves.clear ();

        if (enable_incremetal_mode_) {
          PCL_WARN ("%s: Using kdtree in incremental mode, may affect computation efficiency.", 
            this->getClassName ().c_str ());
          return 0; /// TODO @wgh
        }

        // Check if kdtree has been built
        if (!searchable_)
        {
          PCL_WARN ("%s: Not Searchable", this->getClassName ().c_str ());
          return 0;
        }

        // Find neighbors within radius in the occupied voxel centroid cloud
        std::vector<int> k_indices;
        int k = kdtree_.radiusSearch (point, radius, k_indices, k_sqr_distances, max_nn);

        // Find leaves corresponding to neighbors
        k_leaves.reserve (k);
        for (std::vector<int>::iterator iter = k_indices.begin (); iter != k_indices.end (); iter++)
        {
		  auto leaf = leaves_.find(voxel_centroids_leaf_indices_[*iter]);
		  if (leaf == leaves_.end()) {
			  std::cerr << "error : could not find the leaf corresponding to the voxel" << std::endl;
			  std::cin.ignore(1);
		  }
          k_leaves.push_back (&(leaf->second));
        }
        return k;
      }

      /** \brief Search for all the nearest occupied voxels of the query point in a given radius.
       * \note Only voxels containing a sufficient number of points are used.
       * \param[in] cloud the given query point
       * \param[in] index a valid index in cloud representing a valid (i.e., finite) query point
       * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
       * \param[out] k_leaves the resultant leaves of the neighboring points
       * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
       * \param[in] max_nn
       * \return number of neighbors found
       */
      inline int
      radiusSearch (const PointCloud &cloud, int index, double radius,
                    std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances,
                    unsigned int max_nn = 0) const
      {
        if (index >= static_cast<int> (cloud.points.size ()) || index < 0)
          return (0);
        return (radiusSearch (cloud.points[index], radius, k_leaves, k_sqr_distances, max_nn));
      }

    protected:

      /** \brief Filter cloud and initializes voxel structure.
       * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
       */
      void applyFilter (PointCloud &output);

      /** \brief Flag to determine if voxel structure is searchable. */
      bool searchable_;

      /** \brief Minimum points contained with in a voxel to allow it to be usable. */
      int min_points_per_voxel_;

      /** \brief Minimum allowable ratio between eigenvalues to prevent singular covariance matrices. */
      double min_covar_eigvalue_mult_;

      /** \brief Voxel structure containing all leaf nodes (includes voxels with less than a sufficient number of points). */
	    Map leaves_;

      /** \brief Point cloud containing centroids of voxels containing at least minimum number of points. 
       * [WANG_Guanhua] 所有有效的voxel(点的数量满足阈值)，将voxel几何中心虚拟为点云，可以用于近邻搜索和可视化。
       */
      PointCloudPtr voxel_centroids_;

      /** \brief Indices of leaf structures associated with each point in \ref voxel_centroids_ (used for searching). 
       * [WANG_Guanhua] 实际上指的是点云索引到哈希表key的映射表，也即vector中的element是哈希表中的key。
       */
      std::vector<int> voxel_centroids_leaf_indices_;

      /** \brief KdTree generated using \ref voxel_centroids_ (used for searching). */
      pcl::KdTreeFLANN<PointT> kdtree_;

      // ************************************************************************************************************** 
      // *******************   [WANG_Guanhua] codes below were newly added for incremental version  ******************* 
      // ************************************************************************************************************** 

      /**
       * NOTE: For incremental version, these below variables should be handled carefully, since they get invalid.
       * 
       * ** class VoxelGrid **
       * leaf_layout_
       * save_leaf_layout_
       * 
       * ** class VoxelGridCovariance **
       * leaves_
       * 
       * NOTE: For incremental version, these below variables can still be maintained, with the cost of 
       * lossing computation efficiency.
       * 
       * ** class VoxelGrid **
       * min_b_, max_b_, div_b_, divb_mul_
       * 
       * ** class VoxelGridCovariance **
       * searchable_
       * voxel_centroids_
       * voxel_centroids_leaf_indices_
       * kdtree_
      */

      /**
       * [WANG_Guanhua] We have 3 parameters to determine whether to go over all voxels 
       * to get centroid cloud, they are:
       * 
       * 1. searchable_ : build kdtree or not  NOTE: building kdtree may take some time.
       * 2. enable_obb_update_ : update obb info or not
       * 3. generate_voxel_centroid_cloud_ : generate point cloud from valid voxels or not (for visualization)
       * 
      */

    public:

      /** \brief [WANG_Guanhua] Enable or disable incremental mode.
       * \param[in] value 
       * \return 
       */
      inline void 
        setIncrementalMode(const bool &value) { enable_incremetal_mode_ = value; }

      /** \brief [WANG_Guanhua] When incremental mode enabled, you can voxel filter all target points 
       * within same voxel space, this can avoid too many points being repeatedly inserted into same area.
       * \param[in] value 
       * \param[in] voxel_size 
       * \return 
       * 
       * [TODO:@wgh] for implementation, use a local voxel structure attached to a leaf, but for voxel-dowmsample purpose only.
       */
      inline void 
        setVoxelDownsample(const bool &value, const float &voxel_size = 0.1) 
      { 
        enable_voxel_downsample_ = value;
        if (value) {
          voxel_downsample_size_ = voxel_size;
        }
      }

      /** \brief [WANG_Guanhua] When incremental mode enabled, set a spatial limit for voxels, voxels beyond 
       * this limit should be unloaded.
       * \param[in] box_range 
       * \return 
       */
      inline void 
        setBoundingBoxSize(const Eigen::Vector3f &box_range) 
      {
        bounding_box_size_ = box_range;
      }

      /** \brief [WANG_Guanhua] When incremental mode enabled, update OBB min & max point or not.
       * \param[in] value 
       * \return 
       */
      inline void
        enableObbUpdate(const bool &value) { enable_obb_update_ = value; }

      /** \brief [WANG_Guanhua] When incremental mode enabled, try to trim voxels beyond spatial limit 
       * every several meters.
       * \param[in] every_n_meters 
       * \return 
       */
      inline void 
        setTrimEveryNMeters(const float &every_n_meters) 
      {
        trim_every_n_meters_ = every_n_meters;
      }

      /** \brief [WANG_Guanhua] Get param value.
       * \param[in]  
       * \return param value.
       */
      inline float getTrimEveryNMeters() { return trim_every_n_meters_; }

      /** \brief [WANG_Guanhua] When incremental mode enabled, give current position so that voxels 
       * too far-away from current position can be unloaded.
       * \param[in] position 
       * \return 
       */
      inline void 
        updateVoxelMapCenterAndTrim(const Eigen::Vector3f &position);

      /** \brief [WANG_Guanhua] if true, print debug info, such as mean, cov, eigen value & vectors.
       * \param[in] value 
       * \return 
       */
      inline void 
        showVerboseInfo(const bool &value, const int & level = 0) 
      { 
        print_verbose_info_ = value; 
        if (value) {
          verbose_info_level_ = level >= 0 ? level : -1;
        } else {
          verbose_info_level_ = -1;
        }
      }

    private:

      /** \brief [WANG_Guanhua] Filter cloud and initializes voxel structure.
       * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
       */
      void applyFilterUnderIncrementalMode (PointCloud &output);

    protected:

      /** \brief [WANG_Guanhua] Enable incremental version and block traditional version. */
      bool enable_incremetal_mode_;

      /** \brief [WANG_Guanhua] Enable voxel downsample among different target point clouds. */
      bool enable_voxel_downsample_;
      float voxel_downsample_size_;

      /** \brief [WANG_Guanhua] Voxels beyond this bounding-box should be unloaded from container. */
      Eigen::Vector3f bounding_box_size_;

      /** \brief [WANG_Guanhua] try to trim out-of-box voxels every n meters. */
      float trim_every_n_meters_;
      Eigen::Vector3f current_position_;
      Eigen::Vector3f last_trimmed_position_;

      /** \brief [WANG_Guanhua] Valid voxel controid as pointcloud, then get kdtree; 
       * kdtree for 1d-index, then through below mapping to 3d-key, then to query hash map. 
       */
      std::vector<Eigen::Array3i> voxel_centroids_leaf_3d_indices_;

      /** \brief [WANG_Guanhua] Whether to generate point cloud from valid voxels for visualization. */
      bool generate_voxel_centroid_cloud_;

      /** \brief [WANG_Guanhua] Enable or disable OBB min/max point updating during processing. */
      bool enable_obb_update_;

      /** \brief [WANG_Guanhua] if true, print debug info, such as mean, cov, eigen value & vectors. 
       * level -1: don't print log
       * level  0: print most-valuable log only
       * level  1: print valuable log
       * level  2: print all log
       */
      bool print_verbose_info_;
      int  verbose_info_level_;

      /** \brief [WANG_Guanhua] Carefully designed spacial hash function, required by hash map. */
      struct SpaceHashFunctor {
        inline size_t operator()(const Eigen::Array3i& key) const {
          return size_t(((key[0]) * long(73856093)) 
                        ^ ((key[1]) * long(471943)) 
                        ^ ((key[2]) * long(83492791))) 
              % size_t(1000000000) /*即使是31位，10亿也是完全ok的*/ ;
        }
      };

      /** \brief [WANG_Guanhua] Judge whether two keys are equal, required by hash map. */
      struct EqualWithFunctor {
        inline bool operator()(const Eigen::Array3i& a, const Eigen::Array3i& b) const {
            return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
        }
      };

      /** \brief [WANG_Guanhua] Spacial hash map, used to hold voxels incrementally. */
      using SpaceHashMap = std::unordered_map<
        Eigen::Array3i, 
        typename std::list<std::pair<Eigen::Array3i, std::unique_ptr<Leaf>>>::iterator, 
        SpaceHashFunctor,
        EqualWithFunctor>;

      /** \brief [WANG_Guanhua] containers that hold voxels. */
      SpaceHashMap voxels_map_;
      std::list<std::pair<Eigen::Array3i, std::unique_ptr<Leaf>>> voxels_list_;

  }; // class

} // namespace pclomp

#endif  //#ifndef PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_H_
