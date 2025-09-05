#pragma once

#include "Eigen/Dense"
#include "GlobalConfig/GlobalConfig.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <array>

namespace Element {

    template <int Dim> class NodeDim;
    template <int Dim, int VerticesCount> class FaceDim;
    template <int Dim, int VerticesCount> class CellDim;

    template <typename T>
    using Ptr = std::shared_ptr<T>;

    using Node2DPtr = Ptr<NodeDim<2>>;
    using Node3DPtr = Ptr<NodeDim<3>>;

    using Face2DPtr = Ptr<FaceDim<2, 2>>;
    using Face3DPtr = Ptr<FaceDim<3, 3>>;

    using Cell2DPtr = Ptr<CellDim<2, 3>>; 
    using Cell3DPtr = Ptr<CellDim<3, 4>>; 

    /**
     * @brief base node class template
     * @tparam Dim dimension, -1 means dynamic dimension
     */
    template <int Dim>
    class NodeDim {
    protected:
        using VectorType = std::conditional_t<Dim == -1, 
                                            Eigen::VectorXd,
                                            Eigen::Matrix<double, Dim, 1>>;
        
        VectorType coordinates;
        int index = -1;
        std::vector<int> aff_faces;
        std::vector<int> aff_cells;

    public:
        virtual ~NodeDim() = default;

        NodeDim() {
            if constexpr (Dim != -1) {
                coordinates.setZero();
            }
        }

        explicit NodeDim(const VectorType& coords, int idx = -1) 
            : coordinates(coords), index(idx) {}

        // dimension specific constructor
        template <int D = Dim, typename = std::enable_if_t<D == 2>>
        NodeDim(double x, double y, int idx = -1) : index(idx) {
            coordinates << x, y;
        }

        template <int D = Dim, typename = std::enable_if_t<D == 3>>
        NodeDim(double x, double y, double z, int idx = -1) : index(idx) {
            coordinates << x, y, z;
        }

        int getDimension() const { 
            if constexpr (Dim == -1) return coordinates.size();
            else return Dim;
        }

        int getIndex() const { return index; }
        void setIndex(int idx) { index = idx; }

        const VectorType& getCoordinates() const { return coordinates; }
        
        void setCoordinates(const VectorType& coords) {
            if constexpr (Dim == -1) {
                if (coordinates.size() != coords.size()) {
                    throw std::invalid_argument("Coordinate dimension mismatch");
                }
            }
            coordinates = coords;
        }

        double distanceTo(const NodeDim& other) const {
            if constexpr (Dim == -1) {
                if (coordinates.size() != other.coordinates.size()) {
                    throw std::invalid_argument("Dimension mismatch");
                }
            }
            
            return (coordinates - other.coordinates).norm();
        }

        // dimension specific methods
        template <int D = Dim, typename = std::enable_if_t<D == 2>>
        double cross2D(const NodeDim<2>& other) const {
            return coordinates.x() * other.coordinates.y() - 
                coordinates.y() * other.coordinates.x();
        }

        template <int D = Dim, typename = std::enable_if_t<D == 3>>
        VectorType cross3D(const NodeDim<3>& other) const {
            return coordinates.cross(other.coordinates);
        }

        // coordinate accessors (only for static dimension)
        template <int D = Dim, typename = std::enable_if_t<D != -1>>
        auto x() const { return coordinates[0]; }
        void setX(double value) { coordinates[0] = value; }

        template <int D = Dim, typename = std::enable_if_t<D == 2 || D == 3>>
        auto y() const { return coordinates[1]; }
        void setY(double value) { coordinates[1] = value; }     

        template <int D = Dim, typename = std::enable_if_t<D == 3>>
        auto z() const { return coordinates[2]; }
        void setZ(double value) { coordinates[2] = value; }

        operator Eigen::VectorXd() const { return coordinates; }

        bool operator==(const NodeDim& other) const {
            return coordinates == other.coordinates && index == other.index;
        }

        bool operator!=(const NodeDim& other) const {
            return !(*this == other);
        }

        // Affiliation accessors
        const std::vector<int>& getAffFaces() const { return aff_faces; }
        const std::vector<int>& getAffCells() const { return aff_cells; }
        int getAffFace(int i) const { return aff_faces[i]; }
        int getAffCell(int i) const { return aff_cells[i]; }
        void addAffFace(int face_index) { aff_faces.push_back(face_index); }
        void addAffCell(int cell_index) { aff_cells.push_back(cell_index); }
    };


    /**
     * @brief base face class template
     * @tparam Dim dimension
     * @tparam VerticesCount number of vertices
     */
    template <int Dim, int VerticesCount>
    class FaceDim {
    protected:
        using TypedNodePtr = std::conditional_t<Dim == 2, Node2DPtr, Node3DPtr>;
        alignedVec<TypedNodePtr> vertices;
        int index = -1;
        int boundary_flag = 0;
        int pair_index = -1;
        std::vector<int> adjacent_cells;
        std::vector<int> local_indices;
        
        mutable double measure = -1.0;
        mutable Eigen::VectorXd normal;
        mutable bool normal_cached = false;

        std::vector<int> aff_cells;

    public:
        FaceDim() {
            static_assert((Dim == 2 && VerticesCount == 2) || 
                        (Dim == 3 && VerticesCount >= 3),
                        "Invalid vertices count for dimension");
            
            vertices.reserve(VerticesCount);
            
            adjacent_cells.resize(2, -1);
            local_indices.resize(2, -1);

        }

        explicit FaceDim(const alignedVec<TypedNodePtr>& verts, int idx = -1, int bflag = 0)
            : vertices(verts), index(idx), boundary_flag(bflag) {
            if (verts.size() != VerticesCount) {
                throw std::invalid_argument("Invalid number of vertices");
            }

            this->vertices = verts;
            this->index = idx;
            this->boundary_flag = bflag;
            adjacent_cells = std::vector<int>(2, -1);
            local_indices = std::vector<int>(2, -1);
        }

        virtual ~FaceDim() = default;

        int getDimension() const { return Dim; }
        int getIndex() const { return index; }
        void setIndex(int idx) { index = idx; }
        int getBoundaryFlag() const { return boundary_flag; }
        void setBoundaryFlag(int flag) { boundary_flag = flag; }
        int getPairIndex() const { return pair_index; }
        void setPairIndex(int idx) { pair_index = idx; }
        bool isBoundary() const { return boundary_flag > 0; }

        size_t getNumVertices() const { return vertices.size(); }
        const alignedVec<TypedNodePtr>& getVertices() const { return vertices; }

        const TypedNodePtr& getVertex(size_t i) const {
            if (i >= vertices.size()) {
                throw std::out_of_range("Vertex index out of range");
            }
            return vertices[i];
        }

        const std::vector<int>& getAdjacentCells() const { return adjacent_cells; }
        const std::vector<int>& getLocalIndices() const { return local_indices; }

        void setAdjacentCells(const std::vector<int>& cells) {
            adjacent_cells = cells;
        }

        void setAdjacentCell(size_t i, int cell_index) {
            if (i >= adjacent_cells.size()) {
                throw std::out_of_range("Adjacent cell index out of range");
            }
            adjacent_cells[i] = cell_index;
        }

        void setLocalIndex(size_t i, int local_index) {
            if (i >= local_indices.size()) {
                throw std::out_of_range("Local index out of range");
            }
            local_indices[i] = local_index;
        }

        virtual double getMeasure() const {
            if (measure < 0) {
                if constexpr (Dim == 2) {
                    const auto& v0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
                    measure = (v1.getCoordinates() - v0.getCoordinates()).norm();
                } else if constexpr (Dim == 3) {
                    const auto& v0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
                    const auto& v2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
                    
                    Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                    measure = v1_coords.cross(v2_coords).norm() / 2.0;
                } else {
                    throw std::runtime_error("Can't compute measure for dynamic dimension face.");
                }
            }
            return measure;
        }

        virtual Eigen::VectorXd getNormal() const {
            if (!normal_cached) {
                if constexpr (Dim == 2) {
                    const auto& v0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
                    
                    Eigen::Vector2d dir = v1.getCoordinates() - v0.getCoordinates();
                    normal = Eigen::VectorXd(Eigen::Vector2d(dir.y(), -dir.x()).normalized());
                } else if constexpr (Dim == 3) {
                    const auto& v0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
                    const auto& v2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
                    
                    Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                    normal = Eigen::VectorXd(v1_coords.cross(v2_coords).normalized());
                } else {
                    throw std::runtime_error("Can't compute normal for dynamic dimension face.");
                }
                normal_cached = true;
            }
            return normal;
        }

        bool hasVertex(const TypedNodePtr& node) const {
            // return std::find(vertices.begin(), vertices.end(), node) != vertices.end();
            if (!node) return false;
            for (const auto& vertex : vertices) {
                if (vertex && *vertex == *node) {
                    return true;
                }
            }
            return false;
        }

        bool isPointOnFace(const TypedNodePtr& point, double tolerance = 1e-16) const {
            if constexpr (Dim == 2) {
                const auto& v0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
                const auto& v1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
                const auto& p = *std::static_pointer_cast<NodeDim<2>>(point);
                
                Eigen::Vector2d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                Eigen::Vector2d v2_coords = p.getCoordinates() - v0.getCoordinates();
                double cross = v1_coords(0) * v2_coords(1) - v1_coords(1) * v2_coords(0);
                if (std::abs(cross) > tolerance) return false;

                double dot = v1_coords.dot(v2_coords);
                double len = v1_coords.squaredNorm();
                return dot >= 0 && dot <= len;
            }
            else if constexpr (Dim == 3) {
                const auto& v0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
                const auto& v1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
                const auto& v2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
                const auto& p = *std::static_pointer_cast<NodeDim<3>>(point);
                
                Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                Eigen::Vector3d v3_coords = p.getCoordinates() - v0.getCoordinates();

                double cross = v1_coords.cross(v2_coords).dot(v3_coords);
                if (std::abs(cross) > tolerance) return false;

                Eigen::Vector3d cross_23 = v2_coords.cross(v3_coords);
                Eigen::Vector3d cross_31 = v3_coords.cross(v1_coords);
                double denom = v1_coords.cross(v2_coords).squaredNorm();
                double r = cross_23.dot(v1_coords.cross(v2_coords)) / denom;
                double s = cross_31.dot(v1_coords.cross(v2_coords)) / denom;
                
                return (r >= -tolerance) && (s >= -tolerance) && (r + s <= 1.0 + tolerance);
            }
            else{
                throw std::runtime_error("Can't check if point is on face for dynamic dimension face.");
            }
        }

        bool operator==(const FaceDim& other) const {
            return vertices == other.vertices && index == other.index && boundary_flag == other.boundary_flag;
        }

        bool operator!=(const FaceDim& other) const {
            return !(*this == other);
        }

        // Affiliation accessors
        const std::vector<int>& getAffCells() const { return aff_cells; }
        int getAffCell(int i) const { return aff_cells[i]; }
        void addAffCell(int cell_index) { aff_cells.push_back(cell_index); }

    };

    /**
     * @brief base cell class template
     * @tparam Dim dimension
     * @tparam VerticesCount number of vertices
     */
    template <int Dim, int VerticesCount>
    class CellDim {
    protected:
        using TypedNodePtr = std::conditional_t<Dim == 2, Node2DPtr, Node3DPtr>;
        using TypedFacePtr = std::conditional_t<Dim == 2, Face2DPtr, Face3DPtr>;
        alignedVec<TypedNodePtr> vertices;
        alignedVec<TypedFacePtr> faces;
        int index = -1;
        std::vector<int> adjacent_cells;
        std::vector<Eigen::VectorXd> outward_norm_vec;
        
        mutable double cached_measure = -1.0;
        mutable Eigen::VectorXd cached_centroid;
        mutable bool centroid_cached = false;

    public:
        CellDim() {
            vertices.reserve(VerticesCount);
            
            if constexpr (Dim == 2 && VerticesCount == 3) {
                faces.resize(3);
                adjacent_cells.resize(3, -1);
            } else if constexpr (Dim == 3 && VerticesCount == 4) {
                faces.resize(4);
                adjacent_cells.resize(4, -1);
            }
        }

        explicit CellDim(const alignedVec<TypedNodePtr>& verts, int idx = -1) 
            : vertices(verts), index(idx) {
            if (verts.size() != VerticesCount) {
                throw std::invalid_argument("Invalid number of vertices");
            }
            
            if constexpr (Dim == 2 && VerticesCount == 3) {
                faces.resize(3);
                adjacent_cells.resize(3, -1);
                for (int i = 0; i < 3; ++i) {
                    alignedVec<TypedNodePtr> edge_vertices = {
                        vertices[i], 
                        vertices[(i + 1) % 3]
                    };
                    faces[i] = std::make_shared<FaceDim<2, 2>>(edge_vertices);
                }
            } else if constexpr (Dim == 3 && VerticesCount == 4) {
                faces.resize(4);
                adjacent_cells.resize(4, -1);
                const std::array<std::array<int, 3>, 4> face_indices = {{
                    {0, 1, 2}, {1, 2, 3}, {2, 3, 0}, {3, 0, 1}
                }};
                
                for (int i = 0; i < 4; ++i) {
                    alignedVec<TypedNodePtr> face_vertices = {
                        vertices[face_indices[i][0]],
                        vertices[face_indices[i][1]],
                        vertices[face_indices[i][2]]
                    };
                    faces[i] = std::make_shared<FaceDim<3, 3>>(face_vertices);
                }
            }
            
            computeOutwardNormVec();
        }

        virtual ~CellDim() = default;

        int getDimension() const { return Dim; }
        int getIndex() const { return index; }
        void setIndex(int idx) { index = idx; }

        const alignedVec<TypedNodePtr>& getVertices() const { return vertices; }
        const TypedNodePtr& getVertex(size_t i) const {
            if (i >= vertices.size()) {
                throw std::out_of_range("Vertex index out of range");
            }
            return vertices[i];
        }

        const alignedVec<TypedFacePtr>& getFaces() const { return faces; }
        const TypedFacePtr& getFace(size_t i) const {
            if (i >= faces.size()) {
                throw std::out_of_range("Face index out of range");
            }
            return faces[i];
        }

        void setFaces(const alignedVec<TypedFacePtr>& faces) {
            if (faces.size() != this->faces.size()) {
                throw std::invalid_argument("Number of faces must match");
            }
            this->faces = faces;
        }

        void setFace(int i, const TypedFacePtr& face) { 
            if (i >= faces.size()) {
                throw std::out_of_range("Face index out of range");
            }
            faces[i] = face; 
        }

        size_t getNumVertices() const { return vertices.size(); }
        size_t getNumFaces() const { return faces.size(); }

        const std::vector<int>& getAdjacentCells() const { return adjacent_cells; }
        int getAdjacentCell(size_t i) const {
            if (i >= adjacent_cells.size()) {
                throw std::out_of_range("Adjacent cell index out of range");
            }
            return adjacent_cells[i];
        }

        void setAdjacentCells(const std::vector<int>& cells) {
            if (cells.size() != faces.size()) {
                throw std::invalid_argument("Number of adjacent cells must match number of faces");
            }
            adjacent_cells = cells;
        }

        void setAdjacentCell(size_t i, int cell_index) {
            if (i >= adjacent_cells.size()) {
                throw std::out_of_range("Adjacent cell index out of range");
            }
            adjacent_cells[i] = cell_index;
        }

        virtual double getMeasure() const {
            if (cached_measure < 0) {
                if constexpr (Dim == 2 && VerticesCount == 3) {
                    const auto& n0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<NodeDim<2>>(vertices[2]);

                    Eigen::Vector2d v1 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector2d v2 = n2.getCoordinates() - n0.getCoordinates();
                    cached_measure = std::abs(v1.x() * v2.y() - v1.y() * v2.x()) / 2.0;
                } else if constexpr (Dim == 3 && VerticesCount == 4) {
                    const auto& n0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
                    const auto& n3 = *std::static_pointer_cast<NodeDim<3>>(vertices[3]);

                    Eigen::Vector3d v1 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v2 = n2.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v3 = n3.getCoordinates() - n0.getCoordinates();
                    cached_measure = std::abs(v1.cross(v2).dot(v3)) / 6.0;
                }
            }
            return cached_measure;
        }

        virtual NodeDim<Dim> getCentroid() const {
            if (!centroid_cached) {
                cached_centroid = Eigen::VectorXd::Zero(Dim);
                for (const auto& vertex : vertices) {
                    cached_centroid += std::static_pointer_cast<NodeDim<Dim>>(vertex)->getCoordinates();
                }
                cached_centroid /= vertices.size();
                centroid_cached = true;
            }
            return NodeDim<Dim>(cached_centroid);
        }

        bool isPointInside(const TypedNodePtr& point, double tolerance = 1e-12) const {
            if constexpr (Dim == 2) {
                const auto& n0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
                const auto& n1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
                const auto& n2 = *std::static_pointer_cast<NodeDim<2>>(vertices[2]);
                const auto& p = *std::static_pointer_cast<NodeDim<2>>(point);

                Eigen::Vector2d v0 = n1.getCoordinates() - n0.getCoordinates();
                Eigen::Vector2d v1 = n2.getCoordinates() - n0.getCoordinates();
                Eigen::Vector2d v2 = p.getCoordinates() - n0.getCoordinates();

                double d00 = v0.dot(v0);
                double d01 = v0.dot(v1);
                double d11 = v1.dot(v1);
                double d20 = v2.dot(v0);
                double d21 = v2.dot(v1);

                double denom = d00 * d11 - d01 * d01;
                
                if (std::abs(denom) < tolerance) {
                    return false;
                }
                
                double w = (d11 * d20 - d01 * d21) / denom;
                double v = (d00 * d21 - d01 * d20) / denom;
                double u = 1.0 - v - w;

                return u >= -tolerance && v >= -tolerance && w >= -tolerance;
            }
            else if constexpr (Dim == 3) {
                const auto& n0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
                const auto& n1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
                const auto& n2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
                const auto& n3 = *std::static_pointer_cast<NodeDim<3>>(vertices[3]);
                const auto& p = *std::static_pointer_cast<NodeDim<3>>(point);

                Eigen::Vector3d v0 = n1.getCoordinates() - n0.getCoordinates();
                Eigen::Vector3d v1 = n2.getCoordinates() - n0.getCoordinates();
                Eigen::Vector3d v2 = n3.getCoordinates() - n0.getCoordinates();
                Eigen::Vector3d v3 = p.getCoordinates() - n0.getCoordinates();

                double V = v0.cross(v1).dot(v2);
                if (std::abs(V) < tolerance) {
                    return false; 
                }

                double V0 = v3.cross(v1).dot(v2);
                double V1 = v0.cross(v3).dot(v2);
                double V2 = v0.cross(v1).dot(v3);

                double a = V0 / V;
                double b = V1 / V;
                double c = V2 / V;
                double d = 1.0 - a - b - c;

                return a >= -tolerance && b >= -tolerance && 
                    c >= -tolerance && d >= -tolerance;
            }
            else{
                throw std::runtime_error("Can't check if point is inside cell for dynamic dimension cell.");
            }
        }

        // bool isPointInside(const TypedNodePtr& point, double tolerance = 1e-16) const {
        //     if constexpr (Dim == 2) {
        //         const auto& n0 = *std::static_pointer_cast<NodeDim<2>>(vertices[0]);
        //         const auto& n1 = *std::static_pointer_cast<NodeDim<2>>(vertices[1]);
        //         const auto& n2 = *std::static_pointer_cast<NodeDim<2>>(vertices[2]);
        //         const auto& p = *std::static_pointer_cast<NodeDim<2>>(point);

        //         Eigen::Vector2d v0 = n2.getCoordinates() - n0.getCoordinates();
        //         Eigen::Vector2d v1 = n1.getCoordinates() - n0.getCoordinates();
        //         Eigen::Vector2d v2 = p.getCoordinates() - n0.getCoordinates();

        //         double d00 = v0.dot(v0);
        //         double d01 = v0.dot(v1);
        //         double d11 = v1.dot(v1);
        //         double d20 = v2.dot(v0);
        //         double d21 = v2.dot(v1);

        //         double denom = d00 * d11 - d01 * d01;
        //         double v = (d11 * d20 - d01 * d21) / denom;
        //         double w = (d00 * d21 - d01 * d20) / denom;
        //         double u = 1.0 - v - w;

        //         return u >= -tolerance && v >= -tolerance && w >= -tolerance;
        //     }
        //     else if constexpr (Dim == 3) {
        //         const auto& n0 = *std::static_pointer_cast<NodeDim<3>>(vertices[0]);
        //         const auto& n1 = *std::static_pointer_cast<NodeDim<3>>(vertices[1]);
        //         const auto& n2 = *std::static_pointer_cast<NodeDim<3>>(vertices[2]);
        //         const auto& n3 = *std::static_pointer_cast<NodeDim<3>>(vertices[3]);
        //         const auto& p = *std::static_pointer_cast<NodeDim<3>>(point);

        //         Eigen::Vector3d v0 = n1.getCoordinates() - n0.getCoordinates();
        //         Eigen::Vector3d v1 = n2.getCoordinates() - n0.getCoordinates();
        //         Eigen::Vector3d v2 = n3.getCoordinates() - n0.getCoordinates();
        //         Eigen::Vector3d v3 = p.getCoordinates() - n0.getCoordinates();

        //         double d00 = v0.dot(v0);
        //         double d01 = v0.dot(v1);
        //         double d02 = v0.dot(v2);
        //         double d11 = v1.dot(v1);
        //         double d12 = v1.dot(v2);
        //         double d22 = v2.dot(v2);
        //         double d30 = v3.dot(v0);
        //         double d31 = v3.dot(v1);
        //         double d32 = v3.dot(v2);

        //         double det = d00 * (d11 * d22 - d12 * d12) -
        //                     d01 * (d01 * d22 - d12 * d02) +
        //                     d02 * (d01 * d12 - d11 * d02);

        //         double a = (d30 * (d11 * d22 - d12 * d12) -
        //                 d31 * (d01 * d22 - d12 * d32) +
        //                 d32 * (d01 * d12 - d11 * d30)) / det;

        //         double b = (d00 * (d31 * d22 - d12 * d32) -
        //                 d01 * (d30 * d22 - d12 * d32) +
        //                 d02 * (d30 * d12 - d31 * d02)) / det;

        //         double c = (d00 * (d11 * d32 - d31 * d12) -
        //                 d01 * (d01 * d32 - d31 * d02) +
        //                 d02 * (d01 * d12 - d11 * d30)) / det;

        //         double d = 1.0 - a - b - c;

        //         return a >= -tolerance && b >= -tolerance && 
        //             c >= -tolerance && d >= -tolerance;
        //     }
        //     else{
        //         throw std::runtime_error("Can't check if point is inside cell for dynamic dimension cell.");
        //     }
        // }

        const std::vector<Eigen::VectorXd>& getOutwardNormVec() const { return outward_norm_vec; }

        void computeOutwardNormVec() {
            outward_norm_vec.resize(faces.size());
            auto centroid = getCentroid().getCoordinates();
            
            for (size_t i = 0; i < faces.size(); ++i) {
                Eigen::VectorXd face_normal = faces[i]->getNormal();
                Eigen::VectorXd face_centroid;
                
                if constexpr (Dim == 2) {
                    const auto& face = *std::static_pointer_cast<FaceDim<2, 2>>(faces[i]);
                    const auto& n1 = *std::static_pointer_cast<NodeDim<2>>(face.getVertices()[0]);
                    const auto& n2 = *std::static_pointer_cast<NodeDim<2>>(face.getVertices()[1]);
                    face_centroid = (n1.getCoordinates() + n2.getCoordinates()) / 2.0;
                } else {
                    const auto& face = *std::static_pointer_cast<FaceDim<3, 3>>(faces[i]);
                    face_centroid = Eigen::VectorXd::Zero(3);
                    for (const auto& v : face.getVertices()) {
                        face_centroid += std::static_pointer_cast<NodeDim<3>>(v)->getCoordinates();
                    }
                    face_centroid /= face.getVertices().size();
                }
                
                Eigen::VectorXd cell_to_face = face_centroid - centroid;
                if (face_normal.dot(cell_to_face) < 0) {
                    face_normal = -face_normal;
                }
                
                outward_norm_vec[i] = face_normal;
            }
        }

        bool operator==(const CellDim& other) const {
            return vertices == other.vertices && faces == other.faces && index == other.index;
        }

        bool operator!=(const CellDim& other) const {
            return !(*this == other);
        }
    };

    using Node2D = NodeDim<2>;
    using Node3D = NodeDim<3>;

    using Face2D = FaceDim<2, 2>;
    using Face3D = FaceDim<3, 3>;

    using Cell2D = CellDim<2, 3>;  
    using Cell3D = CellDim<3, 4>;  

} // namespace Element