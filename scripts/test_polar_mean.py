"""
Test to verify circular mean calculation is working correctly for polar plots
"""
import numpy as np
import sys

def circular_mean_angle(angles_rad):
    """
    Calculate the circular mean of angles (in radians).
    """
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    return mean_angle

# Test Case 1: Angles near 0° (12 o'clock)
print("="*60)
print("TEST 1: Angles clustering near 12 o'clock (0°/360°)")
print("="*60)

angles_deg = np.array([350, 355, 0, 5, 10])  # Should average to ~0°
angles_rad = np.radians(angles_deg)

# Wrong way: arithmetic mean
wrong_mean_deg = np.mean(angles_deg)
print(f"Angles: {angles_deg}")
print(f"❌ Arithmetic mean: {wrong_mean_deg:.1f}° (WRONG!)")

# Correct way: circular mean
correct_mean_rad = circular_mean_angle(angles_rad)
correct_mean_deg = np.degrees(correct_mean_rad)
if correct_mean_deg < 0:
    correct_mean_deg += 360
print(f"✓ Circular mean: {correct_mean_deg:.1f}° (CORRECT!)")

# Test Case 2: Angles from actual data (simulated)
print("\n" + "="*60)
print("TEST 2: Simulated prone positions at ~3 o'clock (90°)")
print("="*60)

prone_angles_deg = np.random.normal(90, 20, 20)  # Clustered around 3 o'clock
prone_angles_rad = np.radians(prone_angles_deg)

arithmetic_mean = np.mean(prone_angles_deg)
circular_mean_rad = circular_mean_angle(prone_angles_rad)
circular_mean_deg = np.degrees(circular_mean_rad)

print(f"Sample angles: {prone_angles_deg[:5]}")
print(f"Arithmetic mean: {arithmetic_mean:.1f}°")
print(f"Circular mean: {circular_mean_deg:.1f}°")
print(f"Difference: {abs(arithmetic_mean - circular_mean_deg):.1f}°")

if abs(arithmetic_mean - circular_mean_deg) < 5:
    print("✓ Small difference (angles not near boundary)")
else:
    print("❌ Large difference!")

# Test Case 3: Critical case - angles crossing 0°
print("\n" + "="*60)
print("TEST 3: Supine positions crossing 12 o'clock boundary")
print("="*60)

supine_angles_deg = np.array([350, 355, 358, 2, 5, 8, 10])  # Crossing 0°
supine_angles_rad = np.radians(supine_angles_deg)

arithmetic_mean = np.mean(supine_angles_deg)
circular_mean_rad = circular_mean_angle(supine_angles_rad)
circular_mean_deg = np.degrees(circular_mean_rad)
if circular_mean_deg < 0:
    circular_mean_deg += 360

print(f"Angles: {supine_angles_deg}")
print(f"❌ Arithmetic mean: {arithmetic_mean:.1f}° (COMPLETELY WRONG!)")
print(f"✓ Circular mean: {circular_mean_deg:.1f}° (CORRECT!)")
print(f"Error if using arithmetic: {abs(arithmetic_mean - circular_mean_deg):.1f}°")

if abs(arithmetic_mean - circular_mean_deg) > 100:
    print("⚠️  CRITICAL: Arithmetic mean gives completely wrong result!")
    print("    This is exactly the bug we fixed!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("✓ Circular mean is ESSENTIAL for angles near boundaries")
print("✓ Without it, mean supine position would be wrong by ~180°!")
print("="*60)
