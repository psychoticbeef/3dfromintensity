#ifndef URASTER_HPP
#define URASTER_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>	//for .inverse().  Probably not needed
#include <vector>
#include <array>
#include <memory>
#include <functional>

namespace uraster {

	class Pixel {
		public:
			Eigen::Vector4f color;
			float& depth() {
				return color[3];
			}
			Pixel():color(0.0f,0.0f,0.0f,-1e10f) {
			}
	};

	bool _wireframe = false;


	//This is the framebuffer class.  It's a part of namespace uraster because the uraster needs to have a well-defined image class to render to.
	//It is templated because the output type need not be only colors, could contain anything (like a stencil buffer or depth buffer or gbuffer for deferred rendering)
	template<class PixelType>
		class Framebuffer {
			protected:
				std::vector<PixelType> data;
			public:
				const std::size_t width;
				const std::size_t height;
				//constructor initializes the array
				Framebuffer(std::size_t w,std::size_t h):
					data(w*h),
					width(w),height(h) {
					}
				//2D pixel access
				PixelType& operator()(std::size_t x,std::size_t y) {
					return data[y*width+x];
				}
				//const version
				const PixelType& operator()(std::size_t x,std::size_t y) const {
					return data[y*width+x];
				}
				void clear(const PixelType& pt=PixelType()) {
					std::fill(data.begin(),data.end(),pt);
				}
		};

	//This function runs the vertex shader on all the vertices, producing the varyings that will be interpolated by the rasterizer.
	//VertexVsIn can be anything, VertexVsOut MUST have a position() method that returns a 4D vector, and it must have an overloaded *= and += operator for the interpolation
	//The right way to think of VertexVsOut is that it is the class you write containing the varying outputs from the vertex shader.
	template<class VertexVsIn,class VertexVsOut,class VertShader>
		void run_vertex_shader(const VertexVsIn* b,const VertexVsIn* e,VertexVsOut* o,
				VertShader vertex_shader) {
			std::size_t n=e-b;
#pragma omp parallel for
			for(std::size_t i=0; i<n; i++) {
				o[i]=vertex_shader(b[i]);
			}
		}
	struct BarycentricTransform {
		private:
			Eigen::Vector2f offset;
			Eigen::Matrix2f Ti;
		public:
			BarycentricTransform(const Eigen::Vector2f& s1,const Eigen::Vector2f& s2,const Eigen::Vector2f& s3):
				offset(s3) {
					Eigen::Matrix2f T;
					T << (s1-s3),(s2-s3);
					Ti=T.inverse();
				}
			Eigen::Vector3f operator()(const Eigen::Vector2f& v) const {
				Eigen::Vector2f b;
				b=Ti*(v-offset);
				return Eigen::Vector3f(b[0],b[1],1.0f-b[0]-b[1]);
			}
	};

	void line(int x0, int y0, int x1, int y1, uraster::Framebuffer<uraster::Pixel>& fp, Eigen::Array2i ul, Eigen::Array2i lr)
	{
		int dx =  abs(x1-x0), sx = x0<x1 ? 1 : -1;
		int dy = -abs(y1-y0), sy = y0<y1 ? 1 : -1;
		int err = dx+dy, e2; /* error value e_xy */
		uraster::Pixel p;
		p.color = Eigen::Vector4f(1, 1, 1, 1);

		while(1) {
			if (x0 < ul[0]) x0 = ul[0];
			if (x0 >= lr[0]) x0 = lr[0]-1;
			if (y0 < ul[1]) y0 = ul[1];
			if (y0 >= lr[1]) y0 = lr[1]-1;
			fp(x0, y0) = p;
			//fp(x0+1, y0) = p;
			//fp(x0, y0+1) = p;
			//fp(x0+1, y0+1) = p;
			//fp(x0-1, y0) = p;
			//fp(x0, y0-1) = p;
			//fp(x0-1, y0-1) = p;
			if (x0==x1 && y0==y1) break;
			e2 = 2*err;
			if (e2 > dy) { err += dy; x0 += sx; } /* e_xy+e_x > 0 */
			if (e2 < dx) { err += dx; y0 += sy; } /* e_xy+e_y < 0 */
		}
	}

	//This function takes in 3 varyings vertices from the vertex shader that make up a triangle,
	//rasterizes the triangle and runs the fragment shader on each resulting pixel.
	template<class PixelOut,class VertexVsOut,class FragShader>
		void rasterize_triangle(Framebuffer<PixelOut>& fb,const std::array<VertexVsOut,3>& verts,FragShader fragment_shader) {
			std::array<Eigen::Vector4f,3> points {{verts[0].position(),verts[1].position(),verts[2].position()}};
			//Do the perspective divide by w to get screen space coordinates.
			std::array<Eigen::Vector4f,3> epoints {{points[0]/points[0][3],points[1]/points[1][3],points[2]/points[2][3]}};
			auto ss1=epoints[0].head<2>().array(),ss2=epoints[1].head<2>().array(),ss3=epoints[2].head<2>().array();

			//calculate the bounding box of the triangle in screen space floating point.
			Eigen::Array2f bb_ul=ss1.min(ss2).min(ss3);
			Eigen::Array2f bb_lr=ss1.max(ss2).max(ss3);
			Eigen::Array2i isz(fb.width,fb.height);

			//convert bounding box to fixed point.
			//move bounding box from (-1.0,1.0)->(0,imgdim)
			Eigen::Array2i ibb_ul=((bb_ul*0.5f+0.5f)*isz.cast<float>()).cast<int>();
			Eigen::Array2i ibb_lr=((bb_lr*0.5f+0.5f)*isz.cast<float>()).cast<int>();
			ibb_lr+=1;	//add one pixel of coverage

			//clamp the bounding box to the framebuffer size if necessary
			ibb_ul=ibb_ul.max(Eigen::Array2i(0,0));
			ibb_lr=ibb_lr.min(isz);


			BarycentricTransform bt(ss1.matrix(),ss2.matrix(),ss3.matrix());

				static uraster::Framebuffer<uraster::Pixel> tp(fb.width, fb.height); // mask pixels to draw
			if (_wireframe) {

			int x001 = 0, x010 = 0, x100 = 0;
			int y001 = 0, y010 = 0, y100 = 0;
			Eigen::Vector2f ex_001(1.0, -1.0), ex_010(1.0, -1.0), ex_100(1.0, -1.0);
			// find the extremities in barycentric coordinates of each corner of the
			// vertex in viewport coordinates

			bool visible = false, set_1 = false, set_2 = false, set_3 = false;
			// tell the fragment shader to only draw these in $color
			for(int y=ibb_ul[1]; y<ibb_lr[1]; y++) {
				for(int x=ibb_ul[0]; x<ibb_lr[0]; x++) {
					Eigen::Vector2f ssc(x, y);
					ssc.array()/=isz.cast<float>();	//move pixel to relative coordinates
					ssc.array()-=0.5f;
					ssc.array()*=2.0f;

					//Compute barycentric coordinates of the pixel center
					Eigen::Vector3f bary=bt(ssc);

					//if the pixel has valid barycentric coordinates, the pixel is in the triangle
					if(!((bary.array() <= 1.0f).all() && (bary.array() >= 0.0f).all())) {
						continue;
					}
					visible = true;

					if (bary[0] + bary[1] < ex_001[0] || bary[2] > ex_001[1]) {
					//if (bary[2] > ex_001[1]) {
						ex_001[0] = bary[0] + bary[1];
						ex_001[1] = bary[2];
						x001 = x;
						y001 = y;
						set_1 = true;
					}
					if (bary[0] + bary[2] < ex_010[0] || (bary[1] > ex_010[1])) {
					//if (bary[1] > ex_010[1]) {
						ex_010[0] = bary[0] + bary[2];
						ex_010[1] = bary[1];
						x010 = x;
						y010 = y;
						set_2 = true;
					}
					if (bary[1] + bary[2] < ex_100[0] || (bary[0] > ex_100[1])) {
					//if (bary[0] > ex_100[1]) {
						ex_100[0] = bary[1] + bary[2];
						ex_100[1] = bary[0];
						x100 = x;
						y100 = y;
						set_3 = true;
					}
				}
			}


				tp.clear();
				if (visible && set_1 && set_2 && set_3) {
					line(x001, y001, x010, y010, tp, ibb_ul, ibb_lr);
					line(x010, y010, x100, y100, tp, ibb_ul, ibb_lr);
					line(x100, y100, x001, y001, tp, ibb_ul, ibb_lr);
				}
					}
				//

				//std::cout << coords.size() << " ";

				// calculate all display coordinates of projected edges of the triangle


				//std::cout << ex_001 << std::endl << ex_010 << std::endl << ex_100 << std::endl;
				//exit(1);
				auto a = verts[2].p - verts[1].p;
				auto b = verts[1].p - verts[0].p;
				Eigen::Vector3f c(a[0], a[1], a[2]);
				Eigen::Vector3f d(b[0], b[1], b[2]);
				auto normal = c.cross(d).normalized();

				//for all the pixels in the bounding box
				for(int y=ibb_ul[1]; y<ibb_lr[1]; y++) {
					for(int x=ibb_ul[0]; x<ibb_lr[0]; x++) {
						Eigen::Vector2f ssc(x,y);
						ssc.array()/=isz.cast<float>();	//move pixel to relative coordinates
						ssc.array()-=0.5f;
						ssc.array()*=2.0f;

						//Compute barycentric coordinates of the pixel center
						Eigen::Vector3f bary=bt(ssc);

						//if the pixel has valid barycentric coordinates, the pixel is in the triangle
						if(!((bary.array() < 1.0f).all() && (bary.array() > 0.0f).all())) {
							continue;
						}
						float d=bary[0]*epoints[0][2]+bary[1]*epoints[1][2]+bary[2]*epoints[2][2];
						//Reference the current pixel at that coordinate
						PixelOut& po=fb(x,y);
						// if the interpolated depth passes the depth test
						if(po.depth() >= d || d > 1.0) {
							continue;
						}
						//interpolate varying parameters
						VertexVsOut v;
						for(int i=0; i<3; i++) {
							VertexVsOut vt=verts[i];
							vt*=bary[i];
							v+=vt;
						}
						if (_wireframe) {
						if (tp(x, y).color[0] == 1) {
							v.p[3] = 0.0;
						} else {
							v.p[3] = 0.5;
						}
						}
						v.p[0] = normal[0];
						v.p[1] = normal[1];
						v.p[2] = normal[2];
						//call the fragment shader
						po=fragment_shader(v);
						po.depth()=d; //write the depth buffer
					}
				}
					}

					//This function rasterizes a set of triangles determined by an index buffer and a buffer of output verts.
					template<class PixelOut,class VertexVsOut,class FragShader>
						void rasterize(Framebuffer<PixelOut>& fb,const std::size_t* ib,const std::size_t* ie,const VertexVsOut* verts,
								FragShader fragment_shader) {
							std::size_t n=ie-ib;
#pragma omp parallel for
							for(std::size_t i=0; i<n; i+=3) {
								const std::size_t* ti=ib+i;
								std::array<VertexVsOut,3> tri {{verts[ti[0]],verts[ti[1]],verts[ti[2]]}};
								rasterize_triangle(fb,tri,fragment_shader);
							}
						}

					//This function does a draw call from an indexed buffer
					template<class PixelOut,class VertexVsOut,class VertexVsIn,class VertShader, class FragShader>
						void draw(Framebuffer<PixelOut>& fb,
								const VertexVsIn* vertexbuffer_b,const VertexVsIn* vertexbuffer_e,
								const std::size_t* indexbuffer_b,const std::size_t* indexbuffer_e,
								VertexVsOut* vcache_b,VertexVsOut* vcache_e,
								VertShader vertex_shader,
								FragShader fragment_shader, bool wireframe = false) {
							std::unique_ptr<VertexVsOut[]> vc;
							_wireframe = wireframe;
							if(vcache_b==NULL || (vcache_e-vcache_b) != (vertexbuffer_e-vertexbuffer_b)) {
								vcache_b=new VertexVsOut[(vertexbuffer_e-vertexbuffer_b)];
								vc.reset(vcache_b);
							}
							run_vertex_shader(vertexbuffer_b,vertexbuffer_e,vcache_b,vertex_shader);
							rasterize(fb,indexbuffer_b,indexbuffer_e,vcache_b,fragment_shader);
						}

					struct VertVsOut {
						Eigen::Vector4f p;
						Eigen::Vector3f color;

						VertVsOut():
							p(0.0f,0.0f,0.0f,0.0f),color(0.0f,0.0f,0.0f) {
							}
						const Eigen::Vector4f& position() const {
							return p;
						}
						VertVsOut& operator+=(const VertVsOut& tp) {
							p+=tp.p;
							color+=tp.color;
							return *this;
						}
						VertVsOut& operator*=(const float& f) {
							p*=f;
							color*=f;
							return *this;
						}
					};

					//VertVsOut example_vertex_shader(const Eigen::Vector3f& vin,const Eigen::Matrix4f& mvp,float t) {
					//	VertVsOut vout;
					//	vout.p=mvp*Eigen::Vector4f(vin[0],vin[1],vin[2],1.0f);
					//	//vout.p[3]=1.0f;
					//	vout.color=Eigen::Vector3f(1.0f,static_cast <float> (rand()) / static_cast <float> (RAND_MAX),0.0f);
					//	return vout;
					//}

					//Pixel example_fragment_shader(const VertVsOut& fsin) {
					//	Pixel p;
					//	p.color.head<3>()=fsin.color;
					//	return p;
					//}
					// example end

				}

#endif
