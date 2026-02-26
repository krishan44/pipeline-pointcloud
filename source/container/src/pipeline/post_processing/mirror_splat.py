#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""
Script to mirror a 3D Gaussian Splatting PLY file along a specified axis.
This handles both the mirroring of positions and the transformation of rotations.
"""

import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import plyfile
import sys

def mirror_ply(input_path, output_path=None, axis='x'):
    """
    Mirror a PLY file along the specified axis.
    
    Args:
        input_path: Path to the input PLY file
        output_path: Path to save the mirrored PLY file (if None, overwrites input)
        axis: Axis to mirror along ('x', 'y', or 'z')
    """
    if output_path is None:
        output_path = input_path
    
    # Load the PLY file
    plydata = plyfile.PlyData.read(input_path)
    vertices = plydata['vertex']
    
    # Get all property names
    property_names = [prop.name for prop in vertices.properties]
    
    # Extract data
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rotations = np.vstack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']]).T
    
    # Create mirror matrix
    mirror_matrix = np.eye(3)
    if axis == 'x':
        mirror_matrix[0, 0] = -1
    elif axis == 'y':
        mirror_matrix[1, 1] = -1
    elif axis == 'z':
        mirror_matrix[2, 2] = -1
    
    # Mirror positions
    mirrored_positions = np.dot(positions, mirror_matrix)
    
    # Handle rotations (mirroring changes handedness) in batch
    q_scipy_format = np.column_stack((rotations[:, 1], rotations[:, 2], rotations[:, 3], rotations[:, 0]))
    rot_matrices = Rotation.from_quat(q_scipy_format).as_matrix()
    mirrored_rot_matrices = np.einsum('ij,njk->nik', mirror_matrix, rot_matrices)

    dets = np.linalg.det(mirrored_rot_matrices)
    reflection_mask = dets < 0.0
    if np.any(reflection_mask):
        mirrored_rot_matrices[reflection_mask, :, 0] *= -1.0

    try:
        mirrored_quats_xyzw = Rotation.from_matrix(mirrored_rot_matrices).as_quat()
        mirrored_rotations = np.column_stack((
            mirrored_quats_xyzw[:, 3],
            mirrored_quats_xyzw[:, 0],
            mirrored_quats_xyzw[:, 1],
            mirrored_quats_xyzw[:, 2]
        ))
    except ValueError:
        print("Warning: Failed to convert some mirrored rotation matrices to quaternions. Using original quaternions.")
        mirrored_rotations = rotations.copy()
    
    # Extract spherical harmonic coefficients if they exist
    sh_fields = {'dc': [], 'rest': []}
    
    for field in property_names:
        if field.startswith('f_dc_'):
            sh_fields['dc'].append(field)
        elif field.startswith('f_rest_'):
            sh_fields['rest'].append(field)
    
    # Sort the fields to ensure correct order
    sh_fields['dc'].sort(key=lambda x: int(x.split('_')[-1]))
    sh_fields['rest'].sort(key=lambda x: int(x.split('_')[-1]))
    
    # Extract and mirror spherical harmonic coefficients if they exist
    sh_dc = None
    sh_rest = None
    
    if sh_fields['dc']:
        dc_components = [vertices[field] for field in sh_fields['dc']]
        sh_dc = np.vstack(dc_components).T
        # DC components (RGB values) don't change with mirroring
    
    if sh_fields['rest']:
        rest_components = [vertices[field] for field in sh_fields['rest']]
        sh_rest = np.vstack(rest_components).T
        
        # For degree 1 SH (linear terms), apply the mirror transformation
        if sh_rest.shape[1] >= 9:
            for i in range(0, 9, 3):
                # Extract the 3 coefficients for one color channel
                coeff = sh_rest[:, i:i+3]
                # Apply mirroring
                mirrored_coeff = np.dot(coeff, mirror_matrix)
                # Put back
                sh_rest[:, i:i+3] = mirrored_coeff
    
    # Create new vertices with mirrored data
    properties = [(prop.name, 'f4') for prop in vertices.properties]
    new_vertices = np.zeros(len(vertices), dtype=properties)
    
    # Copy all fields from original data
    for prop_name in property_names:
        if prop_name in ['x', 'y', 'z']:
            continue  # We'll update these separately
        if prop_name.startswith('rot_'):
            continue  # We'll update these separately
        if prop_name.startswith('f_dc_') and sh_dc is not None:
            continue  # We'll update these separately
        if prop_name.startswith('f_rest_') and sh_rest is not None:
            continue  # We'll update these separately
        
        # Copy the original field
        new_vertices[prop_name] = vertices[prop_name]
    
    # Update mirrored positions
    new_vertices['x'] = mirrored_positions[:, 0]
    new_vertices['y'] = mirrored_positions[:, 1]
    new_vertices['z'] = mirrored_positions[:, 2]
    
    # Update mirrored rotations
    new_vertices['rot_0'] = mirrored_rotations[:, 0]
    new_vertices['rot_1'] = mirrored_rotations[:, 1]
    new_vertices['rot_2'] = mirrored_rotations[:, 2]
    new_vertices['rot_3'] = mirrored_rotations[:, 3]
    
    # Update spherical harmonic DC coefficients if they exist
    if sh_dc is not None:
        for i, field in enumerate(sh_fields['dc']):
            new_vertices[field] = sh_dc[:, i]
    
    # Update spherical harmonic rest coefficients if they exist
    if sh_rest is not None:
        for i, field in enumerate(sh_fields['rest']):
            new_vertices[field] = sh_rest[:, i]
    
    # Create PLY element and save
    vertex_element = plyfile.PlyElement.describe(new_vertices, 'vertex')
    ply_data = plyfile.PlyData([vertex_element], comments=plydata.comments)
    ply_data.write(output_path)

def main():
    parser = argparse.ArgumentParser(description='Mirror a Gaussian Splatting PLY file along a specified axis')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input PLY file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output PLY file (default: overwrite input)')
    parser.add_argument('--axis', '-a', choices=['x', 'y', 'z'], default='x', help='Axis to mirror along (default: x)')
    
    args = parser.parse_args()
    
    mirror_ply(args.input, args.output, args.axis)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)