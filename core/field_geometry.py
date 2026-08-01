"""Coordinate transforms for the calibrated Graphwar field."""

X_MIN = -25.0
X_MAX = 25.0
Y_MIN = -15.0
Y_MAX = 15.0


def pixel_to_game(px, py, field_width, field_height):
    """Convert a field-image pixel coordinate to Graphwar coordinates."""
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    game_x = X_MIN + float(px) * (X_MAX - X_MIN) / float(field_width)
    game_y = Y_MAX - float(py) * (Y_MAX - Y_MIN) / float(field_height)
    return game_x, game_y


def game_to_pixel(gx, gy, field_width, field_height):
    """Convert Graphwar coordinates to a field-image pixel coordinate."""
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive")

    px = (float(gx) - X_MIN) * float(field_width) / (X_MAX - X_MIN)
    py = (Y_MAX - float(gy)) * float(field_height) / (Y_MAX - Y_MIN)
    return px, py


def pixel_radius_to_game(radius_px, field_width, field_height):
    """Convert a roughly circular image radius to game units."""
    x_radius = float(radius_px) * (X_MAX - X_MIN) / float(field_width)
    y_radius = float(radius_px) * (Y_MAX - Y_MIN) / float(field_height)
    return (x_radius + y_radius) / 2.0


def pixel_scales(field_width, field_height):
    """Return game units represented by one image pixel on each axis."""
    return (
        (X_MAX - X_MIN) / float(field_width),
        (Y_MAX - Y_MIN) / float(field_height),
    )
