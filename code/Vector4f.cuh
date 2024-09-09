#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class Vector4f
{
public:

	CUDA_CALLABLE Vector4f(float f = 0.f)
	{
		m_elements[0] = f;
		m_elements[1] = f;
		m_elements[2] = f;
		m_elements[3] = f;
	}
	CUDA_CALLABLE Vector4f(float fx, float fy, float fz, float fw)
	{
		m_elements[0] = fx;
		m_elements[1] = fy;
		m_elements[2] = fz;
		m_elements[3] = fw;
	}
	CUDA_CALLABLE Vector4f(float buffer[4])
	{
		m_elements[0] = buffer[0];
		m_elements[1] = buffer[1];
		m_elements[2] = buffer[2];
		m_elements[3] = buffer[3];
	}

	CUDA_CALLABLE Vector4f(const Vector4f& rv)
	{
		m_elements[0] = rv.m_elements[0];
		m_elements[1] = rv.m_elements[1];
		m_elements[2] = rv.m_elements[2];
		m_elements[3] = rv.m_elements[3];
	}

	CUDA_CALLABLE Vector4f& operator = (const Vector4f& rv)
	{
		if (this != &rv)
		{
			m_elements[0] = rv.m_elements[0];
			m_elements[1] = rv.m_elements[1];
			m_elements[2] = rv.m_elements[2];
			m_elements[3] = rv.m_elements[3];
		}
		return *this;
	}

	// no destructor necessary

	// returns the ith element
	CUDA_CALLABLE const float& operator [] (int i) const { return m_elements[i]; }
	CUDA_CALLABLE float& operator [] (int i) { return m_elements[i]; }

	CUDA_CALLABLE float& x() { return m_elements[0]; }
	CUDA_CALLABLE float& y() { return m_elements[1]; }
	CUDA_CALLABLE float& z() { return m_elements[2]; }
	CUDA_CALLABLE float& w() { return m_elements[3]; }

	CUDA_CALLABLE float x() const { return m_elements[0]; }
	CUDA_CALLABLE float y() const { return m_elements[1]; }
	CUDA_CALLABLE float z() const { return m_elements[2]; }
	CUDA_CALLABLE float w() const { return m_elements[3]; }

	CUDA_CALLABLE float length() const
	{
		return sqrt(m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3]);
	}
	CUDA_CALLABLE float lrngthSquared() const
	{
		return(m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3]);
	}
	CUDA_CALLABLE void normalize()
	{
		float norm = sqrt(m_elements[0] * m_elements[0] + m_elements[1] * m_elements[1] + m_elements[2] * m_elements[2] + m_elements[3] * m_elements[3]);
		m_elements[0] = m_elements[0] / norm;
		m_elements[1] = m_elements[1] / norm;
		m_elements[2] = m_elements[2] / norm;
		m_elements[3] = m_elements[3] / norm;
	}
	CUDA_CALLABLE Vector4f normalized() const
	{
		float length = this->length();
		return Vector4f
		(
			m_elements[0] / length,
			m_elements[1] / length,
			m_elements[2] / length,
			m_elements[3] / length
		);
	}

	// if v.z != 0, v = v / v.w
	CUDA_CALLABLE void homogenize()
	{
		if (m_elements[3] != 0)
		{
			m_elements[0] /= m_elements[3];
			m_elements[1] /= m_elements[3];
			m_elements[2] /= m_elements[3];
			m_elements[3] = 1;
		}
	}
	CUDA_CALLABLE Vector4f homogenized() const
	{
		if (m_elements[3] != 0)
		{
			return Vector4f
			(
				m_elements[0] / m_elements[3],
				m_elements[1] / m_elements[3],
				m_elements[2] / m_elements[3],
				1
			);
		}
		else
		{
			return Vector4f
			(
				m_elements[0],
				m_elements[1],
				m_elements[2],
				m_elements[3]
			);
		}
	}

	CUDA_CALLABLE void negate()
	{
		m_elements[0] = -m_elements[0];
		m_elements[1] = -m_elements[1];
		m_elements[2] = -m_elements[2];
		m_elements[3] = -m_elements[3];
	}

	// ---- Utility ----
	CUDA_CALLABLE void print() const
	{
		printf("< %.4f, %.4f, %.4f, %.4f >\n",
			m_elements[0], m_elements[1], m_elements[2], m_elements[3]);
	}

	// static
	CUDA_CALLABLE static float dot(const Vector4f& v0, const Vector4f& v1)
	{
		return v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z() + v0.w() * v1.w();
	}

private:

	float m_elements[4];

};

// component-wise operators
CUDA_CALLABLE inline Vector4f operator + (const Vector4f& v0, const Vector4f& v1)
{
	return Vector4f(v0.x() + v1.x(), v0.y() + v1.y(), v0.z() + v1.z(), v0.w() + v1.w());
}
CUDA_CALLABLE inline Vector4f operator - (const Vector4f& v0, const Vector4f& v1)
{
	return Vector4f(v0.x() - v1.x(), v0.y() - v1.y(), v0.z() - v1.z(), v0.w() - v1.w());
}
CUDA_CALLABLE inline Vector4f operator * (const Vector4f& v0, const Vector4f& v1)
{
	return Vector4f(v0.x() * v1.x(), v0.y() * v1.y(), v0.z() * v1.z(), v0.w() * v1.w());
}
CUDA_CALLABLE inline Vector4f operator / (const Vector4f& v0, const Vector4f& v1)
{
	return Vector4f(v0.x() / v1.x(), v0.y() / v1.y(), v0.z() / v1.z(), v0.w() / v1.w());
}

// unary negation
CUDA_CALLABLE inline Vector4f operator - (const Vector4f& v)
{
	return Vector4f(-v.x(), -v.y(), -v.z(), -v.w());
}

// multiply and divide by scalar
CUDA_CALLABLE inline Vector4f operator * (float f, const Vector4f& v)
{
	return Vector4f(f * v.x(), f * v.y(), f * v.z(), f * v.w());
}
CUDA_CALLABLE inline Vector4f operator * (const Vector4f& v, float f)
{
	return Vector4f(f * v.x(), f * v.y(), f * v.z(), f * v.w());
}
CUDA_CALLABLE inline Vector4f operator / (const Vector4f& v, float f)
{
	return Vector4f(v[0] / f, v[1] / f, v[2] / f, v[3] / f);
}

CUDA_CALLABLE inline bool operator == (const Vector4f& v0, const Vector4f& v1)
{
	return(v0.x() == v1.x() && v0.y() == v1.y() && v0.z() == v1.z() && v0.w() == v1.w());
}
CUDA_CALLABLE inline bool operator != (const Vector4f& v0, const Vector4f& v1)
{
	return !(v0 == v1);
}

