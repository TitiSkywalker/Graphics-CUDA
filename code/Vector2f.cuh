#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include <cmath>
#include <cstdio>

#include "Vector3f.cuh"

class Vector2f
{
public:
	CUDA_CALLABLE Vector2f(float f = 0.f)
	{
		m_elements[0] = f;
		m_elements[1] = f;
	}
	CUDA_CALLABLE Vector2f(float x, float y)
	{
		m_elements[0] = x;
		m_elements[1] = y;
	}

	// copy constructors
	CUDA_CALLABLE Vector2f(const Vector2f& rv)
	{
		m_elements[0] = rv[0];
		m_elements[1] = rv[1];
	}

	// assignment operators
	CUDA_CALLABLE Vector2f& operator = (const Vector2f& rv)
	{
		if (this != &rv)
		{
			m_elements[0] = rv[0];
			m_elements[1] = rv[1];
		}
		return *this;
	}

	// no destructor necessary

	// returns the ith element
	CUDA_CALLABLE const float& operator [] (int i) const { return m_elements[i]; }
	CUDA_CALLABLE float& operator [] (int i) { return m_elements[i]; }

	CUDA_CALLABLE float& x() { return m_elements[0]; }
	CUDA_CALLABLE float& y() { return m_elements[1]; }

	CUDA_CALLABLE float x() const { return m_elements[0]; }
	CUDA_CALLABLE float y() const { return m_elements[1]; }

	CUDA_CALLABLE Vector2f xy() const { return *this; }
	CUDA_CALLABLE Vector2f yx() const { return Vector2f(m_elements[1], m_elements[0]); }
	CUDA_CALLABLE Vector2f xx() const { return Vector2f(m_elements[0], m_elements[0]); }
	CUDA_CALLABLE Vector2f yy() const { return Vector2f(m_elements[1], m_elements[1]); }

	// returns ( -y, x )
	CUDA_CALLABLE Vector2f normal() const { return Vector2f(-m_elements[1], m_elements[0]); }

	CUDA_CALLABLE float length() const { return sqrt(squaredLength()); }
	CUDA_CALLABLE float squaredLength() const { return m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1]; }
	CUDA_CALLABLE void normalize()
	{
		float norm = length();
		m_elements[0] /= norm;
		m_elements[1] /= norm;
	}
	CUDA_CALLABLE Vector2f normalized() const
	{
		float norm = length();
		return Vector2f(m_elements[0] / norm, m_elements[1] / norm);
	}

	CUDA_CALLABLE void negate()
	{
		m_elements[0] = -m_elements[0];
		m_elements[1] = -m_elements[1];
	}

	// ---- Utility ----
	CUDA_CALLABLE void print() const { printf("< %.4f, %.4f >\n", m_elements[0], m_elements[1]); }

	CUDA_CALLABLE Vector2f& operator += (const Vector2f& v)
	{
		m_elements[0] += v.m_elements[0];
		m_elements[1] += v.m_elements[1];
		return *this;
	}
	CUDA_CALLABLE Vector2f& operator -= (const Vector2f& v)
	{
		m_elements[0] -= v.m_elements[0];
		m_elements[1] -= v.m_elements[1];
		return *this;
	}
	CUDA_CALLABLE Vector2f& operator *= (float f)
	{
		m_elements[0] *= f;
		m_elements[1] *= f;
		return *this;
	}

	CUDA_CALLABLE static float dot(const Vector2f& v0, const Vector2f& v1)
	{
		return v0[0] * v1[0] + v0[1] * v1[1];
	}

	CUDA_CALLABLE static Vector3f cross(const Vector2f& v0, const Vector2f& v1)
	{
		return Vector3f
		(
			0,
			0,
			v0.x() * v1.y() - v0.y() * v1.x()
		);
	}

private:

	float m_elements[2];

};

// component-wise operators
CUDA_CALLABLE inline Vector2f operator + (const Vector2f& v0, const Vector2f& v1)
{
	return Vector2f(v0.x() + v1.x(), v0.y() + v1.y());
}
CUDA_CALLABLE inline Vector2f operator - (const Vector2f& v0, const Vector2f& v1)
{
	return Vector2f(v0.x() - v1.x(), v0.y() - v1.y());
}
CUDA_CALLABLE inline Vector2f operator * (const Vector2f& v0, const Vector2f& v1)
{
	return Vector2f(v0.x() * v1.x(), v0.y() * v1.y());
}
CUDA_CALLABLE inline Vector2f operator / (const Vector2f& v0, const Vector2f& v1)
{
	return Vector2f(v0.x() / v1.x(), v0.y() / v1.y());
}

// unary negation
CUDA_CALLABLE inline Vector2f operator - (const Vector2f& v)
{
	return Vector2f(-v.x(), -v.y());
}

// multiply and divide by scalar
CUDA_CALLABLE inline Vector2f operator * (float f, const Vector2f& v)
{
	return Vector2f(f * v.x(), f * v.y());
}
CUDA_CALLABLE inline Vector2f operator * (const Vector2f& v, float f)
{
	return Vector2f(f * v.x(), f * v.y());
}
CUDA_CALLABLE inline Vector2f operator / (const Vector2f& v, float f)
{
	return Vector2f(v.x() / f, v.y() / f);
}

CUDA_CALLABLE inline bool operator == (const Vector2f& v0, const Vector2f& v1)
{
	return(v0.x() == v1.x() && v0.y() == v1.y());
}
CUDA_CALLABLE inline bool operator != (const Vector2f& v0, const Vector2f& v1)
{
	return !(v0 == v1);
}
