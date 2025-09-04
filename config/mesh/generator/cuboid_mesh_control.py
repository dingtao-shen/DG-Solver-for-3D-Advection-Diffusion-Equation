import gmsh
import sys
import math

def generate_cuboid_mesh_with_boundary_refinement(length, width, height, base_mesh_size=1.0, boundary_mesh_size=0.1, refinement_distance=0.2):
    """
    Generate a cuboid mesh with boundary refinement using Gmsh API.
    
    Args:
        length (float): Length of the cuboid along x-axis
        width (float): Width of the cuboid along y-axis
        height (float): Height of the cuboid along z-axis
        base_mesh_size (float): Mesh size in the interior (sparse)
        boundary_mesh_size (float): Mesh size near boundaries (dense)
        refinement_distance (float): Distance from boundary where refinement occurs
    
    Returns:
        None: The mesh is saved to 'cuboid.msh'
    """
    # Initialize Gmsh
    gmsh.initialize()
    
    # Create a new model
    gmsh.model.add("cuboid_with_boundary_refinement")
    
    # Create the cuboid vertices
    p1 = gmsh.model.geo.addPoint(0, 0, 0, boundary_mesh_size)
    p2 = gmsh.model.geo.addPoint(length, 0, 0, boundary_mesh_size)
    p3 = gmsh.model.geo.addPoint(length, width, 0, boundary_mesh_size)
    p4 = gmsh.model.geo.addPoint(0, width, 0, boundary_mesh_size)
    p5 = gmsh.model.geo.addPoint(0, 0, height, boundary_mesh_size)
    p6 = gmsh.model.geo.addPoint(length, 0, height, boundary_mesh_size)
    p7 = gmsh.model.geo.addPoint(length, width, height, boundary_mesh_size)
    p8 = gmsh.model.geo.addPoint(0, width, height, boundary_mesh_size)
    
    # Create the edges
    # Bottom face
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    # Top face
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    
    # Vertical edges
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)
    
    # Create the faces
    # Bottom face
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    bottom = gmsh.model.geo.addPlaneSurface([cl1])
    
    # Top face
    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    top = gmsh.model.geo.addPlaneSurface([cl2])
    
    # Front face
    cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
    front = gmsh.model.geo.addPlaneSurface([cl3])
    
    # Right face
    cl4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
    right = gmsh.model.geo.addPlaneSurface([cl4])
    
    # Back face
    cl5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
    back = gmsh.model.geo.addPlaneSurface([cl5])
    
    # Left face
    cl6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
    left = gmsh.model.geo.addPlaneSurface([cl6])
    
    # Create the volume
    sl = gmsh.model.geo.addSurfaceLoop([bottom, top, front, right, back, left])
    volume = gmsh.model.geo.addVolume([sl])
    
    # Synchronize the model
    gmsh.model.geo.synchronize()
    
    # Define mesh size field for boundary refinement
    # Create a distance field to measure distance from boundaries
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", [bottom, top, front, right, back, left])
    
    # Create a threshold field to control mesh size based on distance
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", boundary_mesh_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", base_mesh_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_distance)
    gmsh.model.mesh.field.setNumber(threshold_field, "StopAtDistMax", 1)
    
    # # Create a math eval field for smoother transition (optional)
    # math_field = gmsh.model.mesh.field.add("MathEval")
    # expr = f"{boundary_mesh_size} + ({base_mesh_size} - {boundary_mesh_size}) * (d/{refinement_distance})^2"
    # gmsh.model.mesh.field.setString(math_field, "F", expr)
    # gmsh.model.mesh.field.setNumber(math_field, "IField", distance_field)
    
    # # Use the math eval field as the background mesh
    # gmsh.model.mesh.field.setAsBackgroundMesh(math_field)
    
    # Alternatively, use the threshold field directly
    # gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # Generate the mesh
    gmsh.model.mesh.generate(3)
    
    # Save the mesh
    gmsh.write("./config/mesh/generator/cuboid_boundary_refined.msh")
    
    # Finalize Gmsh
    gmsh.finalize()

if __name__ == "__main__":
    # Example usage with boundary refinement
    length = 1.0
    width = 1.0
    height = 1.0
    base_mesh_size = 0.4  # Sparse interior
    boundary_mesh_size = 0.2  # Dense boundary
    refinement_distance = 0.2  # Refinement region thickness
    
    generate_cuboid_mesh_with_boundary_refinement(
        length, width, height, base_mesh_size, boundary_mesh_size, refinement_distance
    )
    print(f"Mesh generated successfully with boundary refinement")
    print(f"Interior mesh size: {base_mesh_size}")
    print(f"Boundary mesh size: {boundary_mesh_size}")
    print(f"Refinement distance from boundary: {refinement_distance}")