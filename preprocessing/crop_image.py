from PIL import Image

def crop_image(image, xy):
    coordinates = process_xy(xy)
    corners = get_corners(coordinates)
    square_image = crop_to_square(image, corners)
    square64Image = resize_image(square_image)
    return square64Image

### Helper Functions ##########################
# Extract the polygon pixel coordinates
def process_xy(xy):
    index = xy.index("((")
    return xy[index:].strip("()").split(",")

# Get the corners from the coordinates
def get_corners(coordinates):
    mostLeft = 1024
    mostRight = 0
    mostTop = 1024
    mostBottom = 0
    # Coordinates is a list of coordinates outlining the polygon
    for pair in coordinates:
        tempXY = pair.split()
        x = int(round(float(tempXY[0])))
        y = int(round(float(tempXY[1])))
        if (x < mostLeft):
            mostLeft = x
        if (x > mostRight):
            mostRight = x
        if (y < mostTop):
            mostTop = y
        if (y > mostBottom):
            mostBottom = y
    return (mostLeft, mostRight, mostTop, mostBottom)

# Save the cropped, square matrix of pixels (all dimensions)
def crop_to_square(image, corners):
    (left, right, top, bottom) = corners
    # Find the x_distance and y_distance
    x_distance = right - left
    y_distance = bottom - top

    # Center the image and find the halfLength
    (center_x, center_y, halfLength) = findCenter(corners, x_distance, y_distance)

    # Find the corners
    left = center_x - halfLength
    right = center_x + halfLength
    top = center_y - halfLength
    bottom = center_y + halfLength

    # Adjust if out of bounds
    if (left <= 0):
        left = 0
        right = halfLength * 2
    if (right >= 1024):
        right = 1024
        left = 1024 - halfLength * 2
    if (top <= 0):
        top = 0
        bottom = halfLength * 2
    if (bottom >= 1024):
        bottom = 1024
        top = 1024 - halfLength * 2

    # Return the cropped image
    return image.crop((int(left), int(top), int(right), int(bottom)))

def findCenter(corners, x_distance, y_distance):
    (left, right, top, bottom) = corners
    # Center the image X
    if (x_distance % 2 == 1) :
        halfLength_x = ((x_distance + 1) / 2)
    else :
        halfLength_x = (x_distance / 2)
    center_x = left + halfLength_x
    # Center the image Y
    if (y_distance % 2 == 1) :
        halfLength_y = ((y_distance + 1) / 2)
    else :
        halfLength_y = (y_distance / 2)
    center_y = top + halfLength_y
    # Adjust the halfLength if its's greater than or less than 32
    if (halfLength_x > 32 or halfLength_y > 32):
        if (halfLength_x > halfLength_y):
            halfLength = halfLength_x
        else:
            halfLength = halfLength_y
    else:
        halfLength = 32
    return (center_x, center_y, halfLength)

# If the image is greater than 64 by 64, resize
def resize_image(image):
    if (image.size[0] > 64):
        new_size = (64, 64)
        new_image = image.resize(new_size)
    else:
        new_image = image
    return new_image
