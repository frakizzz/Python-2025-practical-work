def lagrange_interpolation(x_known, y_known, x_target):
    result = 0.0
    n = len(x_known)
    for i in range(n):
        term = 1.0
        for j in range(n):
            if i != j:
                term *= (x_target - x_known[j]) / (x_known[i] - x_known[j])
        result += y_known[i] * term
    return result


if __name__ == "__main__":
    try:
        n = int(input("Enter number of points: "))

        x_known = []
        y_known = []

        print("Enter x and y for each point (separated by space):")
        for i in range(n):
            line = input(f"Point {i + 1}: ").split()
            x_known.append(float(line[0]))
            y_known.append(float(line[1]))

        x_target = float(input("Enter x to interpolate: "))

        interpolated_value = lagrange_interpolation(x_known, y_known, x_target)

        print(f"\nInterpolated value at x={x_target} is {interpolated_value}")

    except ValueError:
        print("Invalid input error")