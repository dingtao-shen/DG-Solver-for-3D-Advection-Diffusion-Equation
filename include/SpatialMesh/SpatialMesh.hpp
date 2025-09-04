#pragma once

#include "Eigen/Dense"
#include "SpatialMesh/Element.hpp"
#include "GlobalConfig/GlobalConfig.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <iomanip> // Required for std::fixed and std::setprecision

namespace SpatialMesh {

    /**
     * @brief base mesh class template
     * @tparam Dim dimension
     */
    template <int Dim>
    class SpatialMeshDim {
        static_assert(Dim == 2 || Dim == 3, "Mesh dimension must be 2 or 3");

        private:
            using TypedNodePtr = std::conditional_t<Dim == 2, Element::Node2DPtr, Element::Node3DPtr>;
            using TypedFacePtr = std::conditional_t<Dim == 2, Element::Face2DPtr, Element::Face3DPtr>;
            using TypedCellPtr = std::conditional_t<Dim == 2, Element::Cell2DPtr, Element::Cell3DPtr>;

            alignedVec<TypedNodePtr> nodes;
            alignedVec<TypedFacePtr> faces;
            alignedVec<TypedCellPtr> cells;
            std::map<int, std::string> boundary_faces;

        public:
            virtual ~SpatialMeshDim() = default;

            SpatialMeshDim() = default;
            explicit SpatialMeshDim(const std::string& filename);

            // Getters
            static constexpr int getDimension() { return Dim; }
            size_t getNumVertices() const { return nodes.size(); }
            size_t getNumFaces() const { return faces.size(); }
            size_t getNumCells() const { return cells.size(); }
            const alignedVec<TypedNodePtr>& getNodes() const { return nodes; }
            const alignedVec<TypedFacePtr>& getFaces() const { return faces; }
            const alignedVec<TypedCellPtr>& getCells() const { return cells; }
            const std::map<int, std::string>& getBoundaryFaces() const { return boundary_faces; }

            std::vector<int> setupComputationOrder(Eigen::VectorXd& Vi);

            // Save mesh information to file
            void SaveMeshInfo(std::string filename) const;

        protected:
            // Mesh connectivity
            virtual void connectFacesToCells();

    };

    template <int Dim>
    SpatialMeshDim<Dim>::SpatialMeshDim(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open mesh file: " + filename);
        }

        std::string line;
        Eigen::MatrixXd vertices_matrix;
        Eigen::MatrixXi elements;
        std::unordered_map<int, int> node_pairs;
        std::vector<std::unordered_map<int, int>> bd_pairs;
        int num_boundary_faces = 0;
        int num_nodes = 0;
        int num_elements = 0;
        int bS, bM, nS, nM, Npairs;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line.find("$PhysicalNames") != std::string::npos) {
                std::getline(file, line);
                num_boundary_faces = std::stoi(line);
                for (int i = 0; i < num_boundary_faces; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int dim, tag;
                    std::string name;
                    iss >> dim >> tag >> name;
                    name = name.substr(1, name.length() - 2);
                    boundary_faces[tag] = name;
                }
                std::getline(file, line); // $EndPhysicalNames
            }
            else if (line.find("$Nodes") != std::string::npos) {
                std::getline(file, line);
                num_nodes = std::stoi(line);
                if(Dim == 2){
                    vertices_matrix.resize(num_nodes, 2);
                }
                else if(Dim == 3){
                    vertices_matrix.resize(num_nodes, 3);
                }
                
                for (int i = 0; i < num_nodes; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int node_id;
                    double x, y, z;
                    iss >> node_id >> x >> y >> z;
                    if(Dim == 2){
                        vertices_matrix.row(node_id - 1) << x, y;
                    }
                    else if(Dim == 3){
                        vertices_matrix.row(node_id - 1) << x, y, z;
                    }
                }
                std::getline(file, line); // $EndNodes
            }
            else if (line.find("$Elements") != std::string::npos) {
                std::getline(file, line); // Read number of elements
                num_elements = std::stoi(line);
                if(Dim == 2){
                    elements.resize(num_elements, 5);
                }
                else if(Dim == 3){
                    elements.resize(num_elements, 6);
                }
                for (int i = 0; i < num_elements; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int elemId, dummy;
                    iss >> elemId;
                    elements.row(elemId - 1).setConstant(-1);
                    iss >> elements(elemId - 1, 0) >> dummy >> elements(elemId - 1, 1) >> dummy;
                    // Read node indices
                    std::vector<int> nodeIndices;
                    int nodeIdx, j = 2;
                    while (iss >> nodeIdx) {
                        elements(elemId - 1, j) = nodeIdx - 1; // Convert to 0-based indexing
                        j++;
                    }
                }
                std::getline(file, line); // Read $EndElements
            }
            else if (line.find("$Periodic") != std::string::npos) {
                int dummy;
                file >> dummy;
                file >> dummy >> bS >> bM;
                bd_pairs.push_back(std::unordered_map<int, int>());
                bd_pairs.back()[bS] = bM;
                bd_pairs.back()[bM] = bS;
                file >> line;
                for(int d = 0; d < 16; d++){file >> dummy;}
                file >> Npairs;
                for(int i = 0; i < Npairs; i++){
                    file >> nS >> nM;
                    node_pairs[nS - 1] = nM - 1;
                    node_pairs[nM - 1] = nS - 1;
                }
                std::getline(file, line); // Read $EndPeriodic
            }
        }

        nodes.clear();
        nodes.reserve(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            Eigen::VectorXd coords = vertices_matrix.row(i);
            nodes.push_back(std::make_shared<Element::NodeDim<Dim>>(
                coords,
                i
            ));
        }

        faces.clear();
        cells.clear();
        
        for (int i = 0; i < num_elements; ++i) {
            const auto& elemRow = elements.row(i);
            int tag = Dim == 2 ? 1 : 2;
            if (elemRow(0) == tag) {
                int boundary_flag = 0;
                if (boundary_faces.find(elemRow(1)) != boundary_faces.end()) {
                    boundary_flag = elemRow(1);
                }
                alignedVec<TypedNodePtr> face_vertices;
                face_vertices.reserve(Dim);
                for(int j = 2; j < 2+Dim; j++){
                    face_vertices.push_back(nodes[elemRow(j)]);
                }
                faces.push_back(std::make_shared<Element::FaceDim<Dim, Dim>>(
                    face_vertices,
                    static_cast<int>(faces.size()),
                    boundary_flag
                ));
            }
        }

        for (int i = 0; i < num_elements; ++i) {
            const auto& elemRow = elements.row(i);
            int tag = Dim == 2 ? 2 : 4;
            if (elemRow(0) == tag) {
                alignedVec<TypedNodePtr> cell_vertices;
                cell_vertices.reserve(Dim+1);
                for(int j = 2; j <= 2+Dim; j++){
                    cell_vertices.push_back(nodes[elemRow(j)]);
                }
                cells.push_back(std::make_shared<Element::CellDim<Dim, Dim+1>>(
                    cell_vertices,
                    static_cast<int>(cells.size())
                ));
                std::vector<std::vector<int>> fv_idx;
                fv_idx.reserve(Dim+1);
                for(int j = 0; j < Dim+1; j++){
                    std::vector<int> fv_idx_;
                    for(int k = 0; k < Dim; k++){
                        fv_idx_.push_back((j+k)%(Dim+1)+2);
                    }
                    fv_idx.push_back(fv_idx_);
                }
                for (const auto& fv : fv_idx) {
                    bool face_exists = false;
                    for (const auto& face : faces) {
                        bool found = true;
                        for(int k = 0; k < Dim; k++){
                            if(!face->hasVertex(nodes[elemRow(fv[k])])){
                                found = false;
                                break;
                            }
                        }
                        if (found) {
                            face_exists = true;
                            break;
                        }
                    }
                    if (!face_exists) {
                        alignedVec<TypedNodePtr> face_vertices;
                        face_vertices.reserve(Dim);
                        for(int k = 0; k < Dim; k++){
                            face_vertices.push_back(nodes[elemRow(fv[k])]);
                        }
                        faces.push_back(std::make_shared<Element::FaceDim<Dim, Dim>>(
                            face_vertices,
                            static_cast<int>(faces.size())
                        ));
                    }
                }
            }
        }

        connectFacesToCells();

        // Populate aff_faces and aff_cells for Node, and aff_cells for Face (3D)
        for (auto& face : faces) {
            for (const auto& node : face->getVertices()) {
                node->addAffFace(face->getIndex());
            }
        }
        for (auto& cell : cells) {
            for (const auto& node : cell->getVertices()) {
                node->addAffCell(cell->getIndex());
            }
            for (const auto& face : cell->getFaces()) {
                face->addAffCell(cell->getIndex());
            }
        }

        // match periodic faces
        if (!node_pairs.empty() && !bd_pairs.empty()) {
            // Create a map to store face pairs based on boundary pairs
            std::unordered_map<int, int> face_pairs;
            
            // For each boundary pair group
            // Find faces that belong to these boundary tags
            std::vector<int> faces_in_group;
            for (size_t i = 0; i < faces.size(); ++i) {
                if (CC.BOUNDARY_COND[faces[i]->getBoundaryFlag()].first == 4) {
                    faces_in_group.push_back(i);
                }
            }
            
            // Match faces based on node pairs
            for (size_t i = 0; i < faces_in_group.size(); ++i) {
                int face1_idx = faces_in_group[i];
                const auto& vertices1 = faces[face1_idx]->getVertices();
                std::set<int> nodes_set1;
                for (const auto& node : vertices1) {
                    int node1_idx = node->getIndex();
                    auto it = node_pairs.find(node1_idx);
                    if (it != node_pairs.end()) {
                        nodes_set1.insert(it->second);
                    }
                    else{
                        throw std::runtime_error("Node " + std::to_string(node1_idx) + " not found in node_pairs");
                    }
                }

                for (size_t j = 0; j < faces_in_group.size(); ++j) {
                    int face2_idx = faces_in_group[j];
                    if (face1_idx == face2_idx) continue;

                    const auto& vertices2 = faces[face2_idx]->getVertices();
                    if (vertices1.size() != vertices2.size()) continue;

                    std::set<int> nodes_set2;
                    for (const auto& node : vertices2) {
                        nodes_set2.insert(node->getIndex());
                    }

                    if (nodes_set1 == nodes_set2){
                        face_pairs[face1_idx] = face2_idx;
                        face_pairs[face2_idx] = face1_idx;
                        faces[face1_idx]->setPairIndex(face2_idx);
                        faces[face2_idx]->setPairIndex(face1_idx);
                        int k;
                        k = (faces[face1_idx]->getAdjacentCells()[0] == -1) ? 0 : 1;
                        faces[face1_idx]->setAdjacentCell(k, faces[face2_idx]->getIndex());
                        k = (faces[face2_idx]->getAdjacentCells()[0] == -1) ? 0 : 1;
                        faces[face2_idx]->setAdjacentCell(k, faces[face1_idx]->getIndex());
                        break;
                    }
                    
                }
            }
        }

        std::cout << "  >>>" << Dim << "D Spatial mesh created successfully" << std::endl;
        std::cout << "  - Nodes: " << nodes.size() << std::endl;
        std::cout << "  - Faces: " << faces.size() << std::endl;
        std::cout << "  - Cells: " << cells.size() << std::endl;
    }

    template <int Dim>
    void SpatialMeshDim<Dim>::connectFacesToCells() {
        int nf = Dim == 2 ? 3 : 4;

        for (size_t i = 0; i < cells.size(); i++) {
            auto cell = cells[i];
            const auto& vertices = cell->getVertices();
            std::vector<int> fv_idx;
            for (size_t j = 0; j < nf; j++) {
                for(size_t k = 0; k < nf-1; k++){
                    fv_idx.push_back((j+k)%nf);
                }
                for (size_t k = 0; k < faces.size(); k++) {
                    auto face = faces[k];
                    bool found = true;
                    for(size_t l = 0; l < nf-1; l++){
                        if(!face->hasVertex(vertices[fv_idx[l]])){
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        cell->setFace(j, face);
                        auto adj_cells = face->getAdjacentCells();
                        size_t pos = adj_cells[0] == -1 ? 0 : 1;
                        face->setAdjacentCell(pos, static_cast<int>(i));
                        face->setLocalIndex(pos, static_cast<int>(j));
                        break;
                    }
                }
                fv_idx.clear();
            }
        }

        for (size_t i = 0; i < cells.size(); i++) {
            auto cell = cells[i];
            std::vector<int> adjacent_cells(nf, -1);
            
            for (size_t j = 0; j < nf; j++) {
                const auto& face = cell->getFace(j);
                const auto& face_adj_cells = face->getAdjacentCells();
                
                for (int adj_cell_idx : face_adj_cells) {
                    if (adj_cell_idx != -1 && adj_cell_idx != static_cast<int>(i)) {
                        adjacent_cells[j] = adj_cell_idx;
                        break;
                    }
                }
            }
            cell->setAdjacentCells(adjacent_cells);
        }

        for (const auto& cell : cells){
            int index = cell->getIndex();
            std::vector<int> adjacent_cells(nf, -1);
            for(size_t i = 0; i < nf; ++i) {
                const auto& face = cell->getFace(i);
                const auto& face_adj_cells = face->getAdjacentCells();
                for (int adj_cell_idx : face_adj_cells) {
                    if (adj_cell_idx != -1 && adj_cell_idx != index) {
                        adjacent_cells[i] = adj_cell_idx;
                        break;
                    }
                }
            }
            cell->setAdjacentCells(adjacent_cells);
        }
    }

    
    template <int Dim>
    std::vector<int> SpatialMeshDim<Dim>::setupComputationOrder(Eigen::VectorXd& Vi) {
        std::vector<int> comp_order_vi(cells.size(), -1);

        // Compute order for each angle

        std::vector<bool> processed(cells.size(), false);
        size_t count = 0;

        while (count < cells.size()) {
            for (size_t i = 0; i < cells.size(); ++i) {
                if (processed[i]) continue;

                bool ready = true;
                const auto& cell = cells[i];
                
                // Check all faces
                for (size_t f = 0; f < cell->getNumFaces(); ++f) {
                    const auto& face = cell->getFace(f);
                    const auto& adj_cells = face->getAdjacentCells();
                    
                    // Find neighbor cell
                    int neighbor_idx = -1;
                    for (int adj : adj_cells) {
                        if (adj != -1 && adj != static_cast<int>(i)) {
                            neighbor_idx = adj;
                            break;
                        }
                    }

                    if (neighbor_idx != -1 && !processed[neighbor_idx]) {
                        const auto& norm = cell->getOutwardNormVec()[f];
                        if (norm.dot(Vi) < 0) {
                            ready = false;
                            break;
                        }
                    }
                }

                if (ready) {
                    comp_order_vi[count++] = i;
                    processed[i] = true;
                }
            }
        }
        return comp_order_vi;
    }


    template <int Dim>
    void SpatialMeshDim<Dim>::SaveMeshInfo(std::string filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open mesh file: " + filename);
        }

        // Write header information
        file << "# ========================================" << std::endl;
        file << "# " << Dim << "D Spatial Mesh Information" << std::endl;
        file << "# Generated by DG-Solver-for-PhononBTE" << std::endl;
        file << "# ========================================" << std::endl;
        file << std::endl;

        // Write mesh statistics
        file << "# Mesh Statistics:" << std::endl;
        file << "# - Total Nodes: " << nodes.size() << std::endl;
        file << "# - Total Faces: " << faces.size() << std::endl;
        file << "# - Total Cells: " << cells.size() << std::endl;
        file << "# - Dimension: " << Dim << "D" << std::endl;
        file << std::endl;

        // Write boundary face information
        file << "# Boundary Face Information:" << std::endl;
        for (const auto& [tag, name] : boundary_faces) {
            file << "# - Boundary " << tag << ": " << name << std::endl;
        }
        file << std::endl;

        // Write detailed cell information in the format matching Test.cpp output
        file << "# ========================================" << std::endl;
        file << "# DETAILED CELL INFORMATION" << std::endl;
        file << "# ========================================" << std::endl;
        file << std::endl;

        for (const auto& cell : cells) {
            int CellID = cell->getIndex();
            file << "CellID = " << CellID << std::endl;
            file << "Centroid = " << cell->getCentroid().getCoordinates().transpose() << std::endl;
            file << "Measure = " << cell->getMeasure() << std::endl;
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            
            // Write vertex information
            for (size_t i = 0; i < cell->getVertices().size(); ++i) {
                const auto& vertex = cell->getVertices()[i];
                file << "Vertices " << vertex->getIndex() << ": " << vertex->getCoordinates().transpose() << std::endl;
            }
            
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            
            // Write outward normal vectors for faces
            const auto& outward_norms = cell->getOutwardNormVec();
            for (size_t i = 0; i < outward_norms.size(); ++i) {
                const auto& face = cell->getFaces()[i];
                file << "Outward Normal for face: " << face->getIndex() << " = " << outward_norms[i].transpose() << std::endl;
            }
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            file << std::endl;
        }

        file << "# ========================================" << std::endl;
        file << "# END OF MESH INFORMATION" << std::endl;
        file << "# ========================================" << std::endl;

        file.close();
        std::cout << "Mesh information saved to: " << filename << std::endl;
    }

} // namespace SpatialMesh