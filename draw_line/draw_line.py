import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class Drawline():
    
    def get_positions_for_specific_track_ids(self, tracks, target_ids):
        specific_position = {}
        for object_name, object_tracks in tracks.items():
            if object_name != "players":
                continue
            for frame_num,frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    if track_id in target_ids and 'position' in track_info:
                        if frame_num not in specific_position:
                            specific_position[frame_num] = {}
                        if track_id not in specific_position[frame_num]:
                            specific_position[frame_num][track_id] = track_info['position']
        return specific_position
    
    def draw_lines_between_tracks(self, frames, specific_positions, transformed_positions, target_ids):
        output_frames = []
    
        for frame_num, frame_positions in specific_positions.items():
            frame = frames[frame_num]
        
            for i in range(len(target_ids) - 1):
                start_id = target_ids[i]
                end_id = target_ids[i+1]
            
                if start_id not in frame_positions or end_id not in frame_positions:
                    continue
            
                start_point = frame_positions[start_id]
                end_point = frame_positions[end_id]
                
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            
                if transformed_positions and frame_num in transformed_positions:
                    if start_id in transformed_positions[frame_num]:
                        start_point_2 = transformed_positions[frame_num][start_id]
                    if end_id in transformed_positions[frame_num]:
                        end_point_2 = transformed_positions[frame_num][end_id]
                        
                distance = np.sqrt((end_point_2[0] - start_point_2[0])**2 + (end_point_2[1] - start_point_2[1])**2)
                
                mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(frame, f"{distance:.2f} m", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)
        
        return output_frames
    
    def transform_positions(self, specific_positions,tracks):
        transformed_positions = {}

        for frame_num, frame_data in specific_positions.items():
            transformed_positions[frame_num] = {} 
            for track_id, position in frame_data.items():
                transformed_positions[frame_num][track_id] = tracks['players'][frame_num][track_id]['for_distance']

        return transformed_positions