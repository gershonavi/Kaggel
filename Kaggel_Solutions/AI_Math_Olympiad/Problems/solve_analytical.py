import sympy
from sympy import symbols, Eq, solve, simplify

def solve_problems():
    print("--- Solving Problem 1: Circle Intersection ---")
    # C1: (x-2)^2 + (y-3)^2 = 25 -> Center (2,3), r1=5
    # C2: (x-8)^2 + (y-11)^2 = 100 -> Center (8,11), r2=10
    
    # Distance between centers
    x1, y1, r1 = 2, 3, 5
    x2, y2, r2 = 8, 11, 10
    
    dist_sq = (x2-x1)**2 + (y2-y1)**2
    dist = sympy.sqrt(dist_sq) # sqrt(6^2 + 8^2) = sqrt(36+64) = 10
    
    print(f"Distance between centers: {dist}")
    print(f"Radius 1: {r1}, Radius 2: {r2}")
    print(f"Sum of radii: {r1+r2}")
    
    # Logic:
    # If dist == r1 + r2 -> Touch externally (1 point)
    # If dist == |r1 - r2| -> Touch internally (1 point)
    # If |r1-r2| < dist < r1+r2 -> Intersect (2 points)
    
    num_intersections = 0
    if dist == r1 + r2 or dist == abs(r1 - r2):
        num_intersections = 1
    elif abs(r1 - r2) < dist < r1 + r2:
        num_intersections = 2
    else:
        num_intersections = 0
        
    print(f"Result Problem 1: {num_intersections}")
    
    print("\n--- Solving Problem 2: Integer Sequence ---")
    # a_{n+1} = a_n / (1 + n*a_n)
    # 1/a_{n+1} = (1 + n*a_n) / a_n = 1/a_n + n
    # Reciprocal form: b_{n+1} = b_n + n, where b_n = 1/a_n
    # b_1 = 1/a_1 = 1
    # b_n = b_1 + sum(1 to n-1)
    # b_n = 1 + (n-1)*n/2
    
    n = 2024
    b_n = 1 + (n-1)*n//2 # Integer division for creating exact int
    a_n = sympy.Rational(1, b_n)
    
    print(f"Calculated b_{n}: {b_n}")
    print(f"Calculated a_{n}: {a_n}")
    
    # LaTeX formatted answer often requires specific format, here we just show value
    return num_intersections, a_n

if __name__ == "__main__":
    solve_problems()
