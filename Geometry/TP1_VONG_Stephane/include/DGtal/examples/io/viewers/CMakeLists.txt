set(DGTAL_TESTS_SRC
   viewer3D-1-points
   viewer3D-2-sets
   viewer3D-3-objects
   viewer3D-4-modes
   viewer3D-5-custom
   viewer3D-4bis-illustrationMode
   viewer3D-6-clipping
   viewer3D-8-2DSliceImages
   viewer3D-8bis-2Dimages
   viewer3D-7bis-planes
   viewer3D-9-3Dimages
   viewer3D-10-interaction
   viewer3D-11-extension
   demo-kernel-2
)

set(DGTAL_TESTS_GMP_SRC
   viewer3D-7-planes
   viewer3D-7-stdplane
   viewer3D-7bis-stdplane
)

foreach(FILE ${DGTAL_TESTS_SRC})
     DGtal_add_example(${FILE})
endforeach()

if(GMP_FOUND)
  foreach(FILE ${DGTAL_TESTS_GMP_SRC})
     DGtal_add_example(${FILE})
  endforeach()
endif()

