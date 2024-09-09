#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

//use this for sqrt function
#include <cmath>

class Vector3f
{
public:
	CUDA_CALLABLE Vector3f(float f = 0.f)
	{
		m_elements[0] = f;
		m_elements[1] = f;
		m_elements[2] = f;
	}
	CUDA_CALLABLE Vector3f(float x, float y, float z)
	{
		m_elements[0] = x;
		m_elements[1] = y;
		m_elements[2] = z;
	}

	// copy constructors
	CUDA_CALLABLE Vector3f(const Vector3f& rv)
	{
		m_elements[0] = rv[0];
		m_elements[1] = rv[1];
		m_elements[2] = rv[2];
	}

	// assignment operators
	CUDA_CALLABLE Vector3f& operator = (const Vector3f& rv)
	{
		if (this != &rv)
		{
			m_elements[0] = rv[0];
			m_elements[1] = rv[1];
			m_elements[2] = rv[2];
		}
		return *this;
	}

	// no destructor necessary

	// returns the ith element
	CUDA_CALLABLE const float& operator [] (int i) const {return m_elements[i];}
	CUDA_CALLABLE float& operator [] (int i) {return m_elements[i];}

	CUDA_CALLABLE float& x() {return m_elements[0];}
	CUDA_CALLABLE float& y() {return m_elements[1];}
	CUDA_CALLABLE float& z() {return m_elements[2];}

	CUDA_CALLABLE float x() const {return m_elements[0];}
	CUDA_CALLABLE float y() const {return m_elements[1];}
	CUDA_CALLABLE float z() const {return m_elements[2];}

	CUDA_CALLABLE Vector3f xyz() const { return Vector3f(m_elements[0], m_elements[1], m_elements[2]); }
	CUDA_CALLABLE Vector3f yzx() const { return Vector3f(m_elements[1], m_elements[2], m_elements[0]); }
	CUDA_CALLABLE Vector3f zxy() const { return Vector3f(m_elements[2], m_elements[0], m_elements[1]); }

	CUDA_CALLABLE float length() const 
	{
		return sqrt(m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2]);
	}
	CUDA_CALLABLE float squaredLength() const
	{
		return
			(
				m_elements[0] * m_elements[0] +
				m_elements[1] * m_elements[1] +
				m_elements[2] * m_elements[2]
				);
	}

	CUDA_CALLABLE void normalize()
	{
		float norm = length();
		m_elements[0] /= norm;
		m_elements[1] /= norm;
		m_elements[2] /= norm;
	}
	CUDA_CALLABLE Vector3f normalized() const
	{
		float norm = length();
		return Vector3f
		(
			m_elements[0] / norm,
			m_elements[1] / norm,
			m_elements[2] / norm
		);
	}

	CUDA_CALLABLE void negate()
	{
		m_elements[0] = -m_elements[0];
		m_elements[1] = -m_elements[1];
		m_elements[2] = -m_elements[2];
	}

	CUDA_CALLABLE void print() const { printf("< %.4f, %.4f, %.4f>\n", m_elements[0], m_elements[1], m_elements[2]); }

	CUDA_CALLABLE Vector3f& operator += (const Vector3f& v)
	{
		m_elements[0] += v.m_elements[0];
		m_elements[1] += v.m_elements[1];
		m_elements[2] += v.m_elements[2];
		return *this;
	}
	CUDA_CALLABLE Vector3f& operator -= (const Vector3f& v)
	{
		m_elements[0] -= v.m_elements[0];
		m_elements[1] -= v.m_elements[1];
		m_elements[2] -= v.m_elements[2];
		return *this;
	}
	CUDA_CALLABLE Vector3f& operator *= (float f)
	{
		m_elements[0] *= f;
		m_elements[1] *= f;
		m_elements[2] *= f;
		return *this;
	}

	CUDA_CALLABLE static float dot(const Vector3f& v0, const Vector3f& v1)
	{
		return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
	}
	CUDA_CALLABLE static Vector3f clamp(const Vector3f& v)
	{
		float x = max(v[0], 0.0f);
		float y = max(v[1], 0.0f);
		float z = max(v[2], 0.0f);

		x = min(v[0], 1.0f);
		y = min(v[1], 1.0f);
		z = min(v[2], 1.0f);

		return Vector3f(x, y, z);
	}
	CUDA_CALLABLE static float clampedDot(const Vector3f& v0, const Vector3f& v1)
	{
		float result = dot(v0, v1);
		result = (result < 0) ? 0 : result;
		result = (result > 1) ? 1 : result;

		return result;
	}

	CUDA_CALLABLE static Vector3f cross(const Vector3f& v0, const Vector3f& v1)
	{
		return Vector3f
		(
			v0.y() * v1.z() - v0.z() * v1.y(),
			v0.z() * v1.x() - v0.x() * v1.z(),
			v0.x() * v1.y() - v0.y() * v1.x()
		);
	}
	CUDA_CALLABLE static Vector3f pointwiseDot(const Vector3f v0, const Vector3f& v1)
	{
		return Vector3f(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]);
	}

private:

	float m_elements[3];

};

// component-wise operators
CUDA_CALLABLE inline Vector3f operator + (const Vector3f& v0, const Vector3f& v1)
{
	return Vector3f(v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2]);
}
CUDA_CALLABLE inline Vector3f operator - (const Vector3f& v0, const Vector3f& v1)
{
	return Vector3f(v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]);
}
CUDA_CALLABLE inline Vector3f operator * (const Vector3f& v0, const Vector3f& v1)
{
	return Vector3f(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]);
}
CUDA_CALLABLE inline  Vector3f operator / (const Vector3f& v0, const Vector3f& v1)
{
	return Vector3f(v0[0] / v1[0], v0[1] / v1[1], v0[2] / v1[2]);
}

// unary negation
CUDA_CALLABLE inline Vector3f operator - (const Vector3f& v)
{
	return Vector3f(-v[0], -v[1], -v[2]);
}

// multiply and divide by scalar
CUDA_CALLABLE inline Vector3f operator * (float f, const Vector3f& v)
{
	return Vector3f(v[0] * f, v[1] * f, v[2] * f);
}
CUDA_CALLABLE inline Vector3f operator * (const Vector3f& v, float f)
{
	return Vector3f(v[0] * f, v[1] * f, v[2] * f);
}
CUDA_CALLABLE inline Vector3f operator / (const Vector3f& v, float f)
{
	return Vector3f(v[0] / f, v[1] / f, v[2] / f);
}

CUDA_CALLABLE inline bool operator == (const Vector3f& v0, const Vector3f& v1)
{
	return(v0.x() == v1.x() && v0.y() == v1.y() && v0.z() == v1.z());
}
CUDA_CALLABLE inline bool operator != (const Vector3f& v0, const Vector3f& v1)
{
	return !(v0 == v1);
}