import base64
import zstd
import struct
import io
import numpy as np
from enum import Enum


class ComponentType(Enum):
    WRITE = (77, 56, 62, 255)
    READ = (46, 71, 93, 255)
    CROSS = (102, 120, 142, 255)
    TUNNEL = (83, 85, 114, 255)
    MESH = (100, 106, 87, 255)
    BUS_0 = (122, 47, 36, 255)
    BUS_1 = (62, 122, 36, 255)
    BUS_2 = (36, 65, 122, 255)
    BUS_3 = (37, 98, 122, 255)
    BUS_4 = (122, 45, 102, 255)
    BUS_5 = (122, 112, 36, 255)
    TC_GRAY = (42, 53, 65, 255)
    TC_WHITE = (159, 168, 174, 255)
    TC_RED = (161, 85, 94, 255)
    TC_ORANGE = (161, 108, 86, 255)
    TC_YELLOW_W = (161, 133, 86, 255)
    TC_YELLOW_C = (161, 152, 86, 255)
    TC_LEMON = (153, 161, 86, 255)
    TC_GREEN_W = (136, 161, 86, 255)
    TC_GREEN_C = (108, 161, 86, 255)
    TC_TURQUOISE = (86, 161, 141, 255)
    TC_BLUE_LIGHT = (86, 147, 161, 255)
    TC_BLUE = (86, 123, 161, 255)
    TC_BLUE_DARK = (86, 98, 161, 255)
    TC_PURPLE = (102, 86, 161, 255)
    TC_VIOLET = (135, 86, 161, 255)
    TC_PINK = (161, 85, 151, 255)
    BUFFER = (146, 255, 99, 255)
    AND = (255, 198, 99, 255)
    OR = (99, 242, 255, 255)
    XOR = (174, 116, 255, 255)
    NOT = (255, 98, 138, 255)
    NAND = (255, 162, 0, 255)
    NOR = (48, 217, 255, 255)
    XNOR = (166, 0, 255, 255)
    LATCH_ON = (99, 255, 159, 255)
    LATCH_OFF = (56, 77, 71, 255)
    CLOCK = (255, 0, 65, 255)
    LED = (255, 255, 255, 255)
    TIMER = (255, 103, 0, 255)
    RANDOM = (229, 255, 0, 255)
    BREAKPOINT = (224, 0, 0, 255)
    WIRELESS_0 = (255, 0, 191, 255)
    WIRELESS_1 = (255, 0, 175, 255)
    WIRELESS_2 = (255, 0, 159, 255)
    WIRELESS_3 = (255, 0, 143, 255)
    ANNOTATION = (58, 69, 81, 255)
    FILLER = (140, 171, 161, 255)
    NONE = (0, 0, 0, 0)


def _readint(buf: io.BytesIO) -> int:
    """Read a 32-bit big-endian unsigned integer from a buffer."""

    return struct.unpack(">I", buf.read(4))[0]


def _read_header(
    buf: io.BytesIO,
) -> tuple[tuple[int, ...], bytes, int, int]:
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


def filter(layer: np.ndarray, component_type: ComponentType) -> np.ndarray:
    """Filters a layer by component type.

    Args:
        layer (np.ndarray): The layer to filter, if shape (m, n, 4).
        component_type (ComponentType): The component type to filter by.

    Returns:
        np.ndarray: The filtered layer. The shape is (m, n).
    """

    # ensure layer is a 3d array of shape (m, n, 4)
    if len(layer.shape) != 3 or layer.shape[2] != 4:
        raise ValueError(
            f"Layer must be a 3d array of shape (m, n, 4). Got shape {layer.shape}."
        )

    return np.all(layer == component_type.value, axis=-1)
