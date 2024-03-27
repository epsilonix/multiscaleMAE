# Converting QPTIFF to Zarr with channel metadata
import os
import xml.etree.ElementTree as ET
import tifffile
import zarr


def qptiff_to_zarr(input_file, output_root, chunk_size=(None, 256, 256)):
    # Check if output file already exists
    output_path = os.path.join(output_root, os.path.basename(input_file).replace('.tif', ''))
    output_zarr = output_path + '/data.zarr'

#    if os.path.exists(output_zarr):
#        print(f'Zarr file already exists at {output_zarr}')
#        return zarr.open(output_zarr, mode='r')
    
    # Create parent directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Read the QPTIFF file
    with tifffile.TiffFile(input_file) as tif:
        for page in tif.pages:
            print(page.shape)
        img_data = tif.asarray()
    xml_file_name = os.path.join(output_path, 'metadata.xml')
    # channel_dict = extract_channel_info_from_qptiff(tif, xml_file_name)
    # Save channels as csv metadata file
    # metadata_file_name = os.path.join(output_path, 'channels.csv')
    # with open(metadata_file_name, 'w') as file:
    #    file.write('channel,marker\n')
    #    for channel, marker in channel_dict.items():
    #        file.write(f'{channel},{marker}\n')
    # Convert to Zarr Array
    z_arr = zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    return z_arr

def extract_channel_info_from_qptiff(tif, xml_file_name) -> dict:
    # Get image description which might contain the metadata
    description = tif.pages[0].description
    # Check if the description is in XML format
    try:
        root = ET.fromstring(description)
        # Save string to xml file for debugging
        with open(xml_file_name, 'w') as f:
             f.write(description)
    except ET.ParseError:
        raise ValueError('Image description is not in XML format')
    # Convert XML to dictionary
    xml_dict = xml_to_dict(root)
    channels_list = extract_channels(xml_dict)
    # Extracting channel names in their original order
    channels_ordered = [channel.get("MarkerName") for channel in channels_list if channel.get("MarkerName")]
    # Creating a dictionary with channel indices and names
    ordered_channel_dict = {index: channel for index, channel in enumerate(channels_ordered, 1)}
    # Removing channels with the name "Blank" and "Empty"
    filtered_channel_dict = {index: channel for index, channel in ordered_channel_dict.items() if channel not in ["Blank", "Empty"]}
    # Updating the indices to be sequential after removing "Blank" and "Empty" channels
    channel_dict = {new_index: channel for new_index, (_, channel) in enumerate(filtered_channel_dict.items())}
    return channel_dict


def extract_channels(data, key="Channels"):
    """Recursively search for channels in a nested dictionary."""
    channels = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key and isinstance(v, dict) and "Channel" in v:
                channels.extend(v["Channel"])
            else:
                channels.extend(extract_channels(v))
    elif isinstance(data, list):
        for item in data:
            channels.extend(extract_channels(item))
    return channels


def xml_to_dict(element):
    """Convert an XML element to a dictionary."""
    # Base case: if the element has no children, return its text
    if len(element) == 0:
        return element.text
    # Recursive case: iterate over child elements and build the dictionary
    elem_dict = {}
    for child in element:
        child_data = xml_to_dict(child)
        # Handle elements with the same tag name by creating a list
        if child.tag in elem_dict:
            if not isinstance(elem_dict[child.tag], list):
                elem_dict[child.tag] = [elem_dict[child.tag]]
            elem_dict[child.tag].append(child_data)
        else:
            elem_dict[child.tag] = child_data
    return elem_dict
