#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include <cstdio>

#include "Vector2f.cuh"

// 2x2 Matrix, stored in column major order (OpenGL style)
class Matrix2f
{
public:

	// Fill a 2x2 matrix with "fill", default to 0.
	CUDA_CALLABLE Matrix2f(float fill = 0.f)
	{
		for (int i = 0; i < 4; ++i)
		{
			m_elements[i] = fill;
		}
	}
	CUDA_CALLABLE Matrix2f(float m00, float m01,
		float m10, float m11)
	{
		m_elements[0] = m00;
		m_elements[1] = m10;

		m_elements[2] = m01;
		m_elements[3] = m11;
	}

	// setColumns = true ==> sets the columns of the matrix to be [v0 v1]
	// otherwise, sets the rows
	CUDA_CALLABLE Matrix2f(const Vector2f& v0, const Vector2f& v1, bool setColumns = true)
	{
		if (setColumns)
		{
			setCol(0, v0);
			setCol(1, v1);
		}
		else
		{
			setRow(0, v0);
			setRow(1, v1);
		}
	}

	CUDA_CALLABLE Matrix2f(const Matrix2f& rm)
	{
		m_elements[0] = rm.m_elements[0];
		m_elements[1] = rm.m_elements[1];
		m_elements[2] = rm.m_elements[2];
		m_elements[3] = rm.m_elements[3];
	}
	CUDA_CALLABLE Matrix2f& operator = (const Matrix2f& rm)
	{
		if (this != &rm)
		{
			m_elements[0] = rm.m_elements[0];
			m_elements[1] = rm.m_elements[1];
			m_elements[2] = rm.m_elements[2];
			m_elements[3] = rm.m_elements[3];
		}

		return *this;
	}
	// no destructor necessary

	CUDA_CALLABLE const float& operator () (int i, int j) const
	{
		return m_elements[j * 2 + i];
	}
	CUDA_CALLABLE float& operator () (int i, int j)
	{
		return m_elements[j * 2 + i];
	}

	CUDA_CALLABLE Vector2f getRow(int i) const
	{
		return Vector2f
		(
			m_elements[i],
			m_elements[i + 2]
		);
	}
	CUDA_CALLABLE void setRow(int i, const Vector2f& v)
	{
		m_elements[i] = v.x();
		m_elements[i + 2] = v.y();
	}

	CUDA_CALLABLE Vector2f getCol(int j) const
	{
		int colStart = 2 * j;

		return Vector2f
		(
			m_elements[colStart],
			m_elements[colStart + 1]
		);
	}
	CUDA_CALLABLE void setCol(int j, const Vector2f& v)
	{
		int colStart = 2 * j;

		m_elements[colStart] = v.x();
		m_elements[colStart + 1] = v.y();
	}

	CUDA_CALLABLE float determinant()
	{
		return Matrix2f::determinant2x2
		(
			m_elements[0], m_elements[2],
			m_elements[1], m_elements[3]
		);
	}
	CUDA_CALLABLE Matrix2f inverse(bool* pbIsSingular = NULL, float epsilon = 0.f)
	{
		float determinant = m_elements[0] * m_elements[3] - m_elements[2] * m_elements[1];

		bool isSingular = (fabs(determinant) < epsilon);
		if (isSingular)
		{
			if (pbIsSingular != NULL)
			{
				*pbIsSingular = true;
			}
			return Matrix2f();
		}
		else
		{
			if (pbIsSingular != NULL)
			{
				*pbIsSingular = false;
			}

			float reciprocalDeterminant = 1.0f / determinant;

			return Matrix2f
			(
				m_elements[3] * reciprocalDeterminant, -m_elements[2] * reciprocalDeterminant,
				-m_elements[1] * reciprocalDeterminant, m_elements[0] * reciprocalDeterminant
			);
		}
	}

	CUDA_CALLABLE void transpose()
	{
		float m01 = (*this)(0, 1);
		float m10 = (*this)(1, 0);

		(*this)(0, 1) = m10;
		(*this)(1, 0) = m01;
	}
	CUDA_CALLABLE Matrix2f transposed() const
	{
		return Matrix2f
		(
			(*this)(0, 0), (*this)(1, 0),
			(*this)(0, 1), (*this)(1, 1)
		);

	}

	// ---- Utility ----
	CUDA_CALLABLE void print()
	{
		printf("[ %.4f %.4f ]\n[ %.4f %.4f ]\n",
			m_elements[0], m_elements[2],
			m_elements[1], m_elements[3]);
	}

	// static
	CUDA_CALLABLE static float determinant2x2(float m00, float m01, float m10, float m11)
	{
		return(m00 * m11 - m01 * m10);
	}

	// static
	CUDA_CALLABLE static Matrix2f ones()
	{
		Matrix2f m;
		for (int i = 0; i < 4; ++i)
		{
			m.m_elements[i] = 1;
		}

		return m;
	}
	// static
	CUDA_CALLABLE static Matrix2f identity()
	{
		Matrix2f m;

		m(0, 0) = 1;
		m(1, 1) = 1;

		return m;
	}
	// static
	CUDA_CALLABLE static Matrix2f rotation(float degrees)
	{
		float c = cos(degrees);
		float s = sin(degrees);

		return Matrix2f
		(
			c, -s,
			s, c
		);
	}

private:

	float m_elements[4];

};

// Scalar-Matrix multiplication
CUDA_CALLABLE inline Matrix2f operator * (float f, const Matrix2f& m)
{
	Matrix2f output;

	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			output(i, j) = f * m(i, j);
		}
	}

	return output;
}
CUDA_CALLABLE inline Matrix2f operator * (const Matrix2f& m, float f)
{
	return f * m;
}

// Matrix-Vector multiplication
// 2x2 * 2x1 ==> 2x1
CUDA_CALLABLE inline Vector2f operator * (const Matrix2f& m, const Vector2f& v)
{
	Vector2f output(0, 0);

	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			output[i] += m(i, j) * v[j];
		}
	}

	return output;
}

// Matrix-Matrix multiplication
CUDA_CALLABLE inline Matrix2f operator * (const Matrix2f& x, const Matrix2f& y)
{
	Matrix2f product; // zeroes

	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			for (int k = 0; k < 2; ++k)
			{
				product(i, k) += x(i, j) * y(j, k);
			}
		}
	}

	return product;
}