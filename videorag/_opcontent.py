import re
import json
import os

from .base import (
    QueryParam
)




async def _refine_visual_sentence_segmentation_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    # Split the text by the section symbol '/////\n'
    # This pattern matches the separator as shown in the examples
    segments = re.split(r'/////\n', query)
    
    # Remove empty segments and strip whitespace
    segments = [segment.strip() for segment in segments if segment.strip()]
    
    # Join the segments with semicolons
    return segments




async def videorag_query(
    query,
    video_segment_feature_vdb,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    query = query
    
    # visual retrieval
    scene_sentences = await _refine_visual_sentence_segmentation_retrieval_query(
        query,
        query_param,
        global_config,
    )

    visual_retrieved_segments = []  # List to store segments for each scene
    used_segments = set()  # Set to track already used segments across all scenes

    

    # Define paths relative to current working directory
    current_dir = os.getcwd()
    dataset_dir = os.path.join(current_dir, 'dataset')
    video_edit_dir = os.path.join(dataset_dir, 'video_edit')
    scene_output_dir = os.path.join(video_edit_dir, 'scene_output')
    working_dir = os.path.join(video_edit_dir, 'videosource-workdir')

    # Replace the existing file operations with these:
    with open(os.path.join(scene_output_dir, 'textual_segmentations.json'), 'w', encoding ='utf-8') as f:
        json.dump(scene_sentences, f)

    with open(os.path.join(working_dir, 'kv_store_video_segments.json'), 'r', encoding ='utf-8') as file:
        kvdata = json.load(file)


    # Calculate segments to exclude for each movie source
    movie_segments_info = {}
    segment_duration = 30  # Default segment duration of 30 seconds


    # Try to determine segment duration from the first movie's first segment
    if kvdata:
        for movie_name, segments in kvdata.items():
            if segments and "0" in segments:
                # Get the time string (e.g., "0-30" or "0-10")
                time_str = segments["0"].get("time", "")
                if time_str and "-" in time_str:
                    start, end = time_str.split("-")
                    try:
                        # Calculate duration from the difference
                        segment_duration = int(end) - int(start)
                        print(f"Detected segment duration: {segment_duration} seconds")
                        break
                    except ValueError:
                        print(f"Could not parse time range '{time_str}', using default duration of 30 seconds")
    
    # Process each movie source
    for movie_id, segments in kvdata.items():
        # Get number of segments for this movie
        num_segments = len(segments)
        
        # Calculate total runtime for this specific movie
        total_runtime = segment_duration * num_segments
        
        # Calculate skip ranges for this movie 
        start_credits_end = max(1, int((total_runtime * 0.1) // segment_duration))
        end_credits_start = min(num_segments - 1, int((total_runtime * 0.90) // segment_duration))
        
        # Store the valid segment range for this video
        movie_segments_info[movie_id] = {
            "total_segments": num_segments,
            "valid_range": (start_credits_end, end_credits_start)
        }
    
    print("Opening/Ending video segments have been filtered for each source video")
    print("Searching for videos...")

    # Process each scene sentence
    for scene in scene_sentences:
        # Query video segments based on scene description
        segment_results = await video_segment_feature_vdb.query(scene)
        
        if len(segment_results):
            # Get more results to have alternatives if top ones are already used
            top_results = segment_results[:20]  # Increased to have more options
            
            # Try each result in order, skipping those already used
            selected_segment = None
            
            for result in top_results:
                segment_id = result['__id__']
                
                # Skip if segment has already been used in previous scenes
                if segment_id in used_segments:
                    continue
                
                # Parse segment ID to extract movie_id and segment_num
                parts = segment_id.split("_")
                
                # Check if segment ID is valid and within valid range for its source movie
                is_valid_segment = False
                
                if len(parts) >= 2:
                    # Extract movie_id and segment_number based on segment_id format
                    if len(parts) == 3:  # Format: movie_video_id_section
                        movie_id = parts[0]
                        try:
                            segment_num = int(parts[2])
                            is_valid_format = True
                        except ValueError:
                            is_valid_format = False
                    elif len(parts) == 2:  # Format: movie_section
                        movie_id = parts[0]
                        try:
                            segment_num = int(parts[1])
                            is_valid_format = True
                        except ValueError:
                            is_valid_format = False
                    else:
                        is_valid_format = False
                        
                    # Check if the segment is within the valid range for its movie
                    if is_valid_format and movie_id in movie_segments_info:
                        valid_range = movie_segments_info[movie_id]["valid_range"]
                        if valid_range[0] <= segment_num <= valid_range[1]:
                            is_valid_segment = True
                
                # If segment is valid, select it
                if is_valid_segment:
                    selected_segment = segment_id
                    break  # Found a valid segment, stop looking
            
            if selected_segment:
                # Mark as used so it won't be selected for future scenes
                used_segments.add(selected_segment)
                visual_retrieved_segments.append([selected_segment])
            else:
                # If no suitable segment found, append empty list
                visual_retrieved_segments.append([])
        else:
            # If no segments found, append empty list
            visual_retrieved_segments.append([])
    

    query_for_visual_retrieval = scene_sentences
    
 




    

    visual_retrieved_segments = [
        segment 
        for segments in visual_retrieved_segments 
        for segment in segments[:1]
    ]




    print(f"Number of scene sentences: {len(scene_sentences)}")
    for i, (scene, segments) in enumerate(zip(scene_sentences, visual_retrieved_segments)):
        print(f"\nScene {i+1}: {scene}")
        print(f"Retrieved Visual Segment: {segments}")

    # Combine all segments while maintaining uniqueness
    retrieved_segments = list(set(visual_retrieved_segments))



    # Sort the segments if needed
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]),  # video_name
            eval(x.split('_')[-1])  # index
        )
    )
    

    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    with open(os.path.join(scene_output_dir, 'visual_retrieved_segments.json'), 'w') as f:
        json.dump(visual_retrieved_segments, f)
    

    return "Search Complete"