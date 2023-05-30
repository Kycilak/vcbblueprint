import base64
import zstd
import struct
import io
import numpy as np


def _readint(buf: io.BytesIO) -> int:
    """Read a 32-bit big-endian unsigned integer from a buffer."""

    return struct.unpack(">I", buf.read(4))[0]


def _read_header(
    buf: io.BytesIO,
) -> tuple[tuple[int, int, int], bytes, int, int]:
    """Reads the header of a Visual Circuit Board blueprint.

    Args:
        buf (io.BytesIO): The buffer to read from.

    Returns:
        tuple[tuple[int, int, int], bytes, int, int]: A tuple containing the
            version, the checksum, the width and the height.
    """

    version = tuple(buf.read(3))
    checksum = buf.read(6)
    width = _readint(buf)
    height = _readint(buf)

    return version, checksum, width, height


def _read_block(buf: io.BytesIO) -> tuple[int, bytes]:
    """Reads a block from a Visual Circuit Board blueprint.

    Args:
        buf (io.BytesIO): The buffer to read from.

    Raises:
        ValueError: If the decompressed data size does not match the size
            specified in the block header.

    Returns:
        tuple[int, bytes]: A tuple containing the layer ID and the
            decompressed data.
    """
    block_size = _readint(buf)
    layer_id = _readint(buf)
    buffer_size = _readint(buf)

    compressed_data = buf.read(block_size - 12)
    uncompressed_data = zstd.decompress(compressed_data)

    if len(uncompressed_data) != buffer_size:
        raise ValueError(
            f"Decompressed data size does not match the expected size. Expected {buffer_size} bytes, got {len(compressed_data)}. The blueprint may be corrupt."
        )

    return layer_id, uncompressed_data


def read_blueprint(
    blueprint_string: str,
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    """Reads a Visual Circuit Board blueprint string and returns a dictionary
    of layers.

    Args:
        blueprint_string (str): The VCB blueprint string.

    Raises:
        ValueError: If the blueprint string does not have the correct
            identifier.

    Returns:
        tuple[tuple[int, ...], dict[int, np.ndarray]]: A tuple containing the
            version and a dictionary of layers. The version is a tuple of
            integers. The dictionary maps layer IDs to RGBA arrays.
            The RGBA arrays are of shape (height, width, 4).
    """

    if len(blueprint_string) < 40:
        raise ValueError("Blueprint string is too short.")

    # check file identifier
    blueprint_identifier = blueprint_string[0:4]
    if blueprint_identifier != "VCB+":
        raise ValueError(
            f"Invalid blueprint identifier. Expected 'VCB+', got '{blueprint_identifier}'."
        )

    # decode base64
    decoded_string = base64.b64decode(blueprint_string[4:])
    buf = io.BytesIO(decoded_string)

    # read header
    version, checksum, width, height = _read_header(buf)

    # TODO: Check checksum

    # read blocks
    layers = {}
    while buf.tell() < len(decoded_string):
        layer_id, uncompressed_data = _read_block(buf)

        rgba = np.frombuffer(uncompressed_data, dtype=np.uint8).reshape(
            (height, width, 4)
        )

        layers[layer_id] = rgba

    return version, layers
