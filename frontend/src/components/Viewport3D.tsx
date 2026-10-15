// Interactive 3D viewport.
//
// Fetches the generated GLB from the backend and renders it with three.js. The
// mesh geometry comes straight from marching cubes on the real segmentation, so
// what is displayed matches the exported model exactly.

import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box3,
  Color,
  Mesh,
  MeshStandardMaterial,
  Vector3,
  type Object3D,
} from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { api } from "../api/client";
import type { MeshResult } from "../types";
import { Badge, Button, Empty } from "./ui";

function Model({
  url,
  color,
  wireframe,
  onFramed,
}: {
  url: string;
  color: string;
  wireframe: boolean;
  onFramed: (center: Vector3, radius: number) => void;
}) {
  const [scene, setScene] = useState<Object3D | null>(null);

  useEffect(() => {
    let disposed = false;
    const loader = new GLTFLoader();
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`GLB request failed (${response.status})`);
        return response.arrayBuffer();
      })
      .then(
        (buffer) =>
          new Promise<void>((resolve, reject) => {
            loader.parse(
              buffer,
              "",
              (gltf) => {
                if (disposed) {
                  resolve();
                  return;
                }
                const box = new Box3().setFromObject(gltf.scene);
                const center = box.getCenter(new Vector3());
                const size = box.getSize(new Vector3());
                const radius = Math.max(size.length() * 0.5, 1);
                setScene(gltf.scene);
                onFramed(center, radius);
                resolve();
              },
              (err) => reject(err),
            );
          }),
      )
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, [url, onFramed]);

  useEffect(() => {
    if (!scene) return;
    scene.traverse((object) => {
      const mesh = object as Mesh;
      if (!mesh.isMesh) return;
      const material = new MeshStandardMaterial({
        color: new Color(color),
        metalness: 0.1,
        roughness: 0.62,
        wireframe,
        flatShading: false,
      });
      mesh.material = material;
      mesh.geometry.computeVertexNormals?.();
    });
  }, [scene, color, wireframe]);

  if (!scene) return null;
  return <primitive object={scene} />;
}

function CameraRig({ center, radius, resetKey }: { center: Vector3; radius: number; resetKey: number }) {
  const controlsRef = useRef<any>(null);
  // Position the camera along a three-quarter anatomical view.
  const position = useMemo<[number, number, number]>(() => {
    const distance = radius * 2.6;
    return [center.x + distance * 0.7, center.y + distance * 0.55, center.z + distance];
  }, [center, radius]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (controls) {
      controls.target.copy(center);
      controls.update();
    }
  }, [center, resetKey]);

  return (
    <>
      <PerspectiveCamera makeDefault position={position} near={radius / 100} far={radius * 50} />
      <OrbitControls ref={controlsRef} makeDefault target={center} enableDamping />
    </>
  );
}

export function Viewport3D({ mesh }: { mesh: MeshResult | null }) {
  const [wireframe, setWireframe] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [framing, setFraming] = useState<{ center: Vector3; radius: number }>({
    center: new Vector3(0, 0, 0),
    radius: 120,
  });

  const url = mesh ? api.meshGlbUrl(mesh.mesh_id) : null;

  if (!mesh || !url) {
    return <Empty>Generate a 3D model from a segmentation to view the surface here.</Empty>;
  }

  const meshMetrics = (mesh.measurements?.mesh ?? {}) as Record<string, unknown>;
  const surfaceArea = meshMetrics.surface_area_mm2;
  const watertight = meshMetrics.is_watertight;

  const handleFramed = (center: Vector3, radius: number) => setFraming({ center, radius });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-1 pb-2">
        <div className="flex items-center gap-2">
          <Badge tone="accent">3D surface</Badge>
          <span className="font-mono text-[11px] text-slate-400">
            {mesh.face_count.toLocaleString()} triangles
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => setWireframe((value) => !value)}>
            {wireframe ? "Shaded" : "Wireframe"}
          </Button>
          <Button onClick={() => setResetKey((key) => key + 1)}>Reset view</Button>
        </div>
      </div>

      <div className="viewport-grid relative min-h-0 flex-1 overflow-hidden rounded-md border border-ink-700/60 bg-ink-950">
        <Canvas>
          <color attach="background" args={["#05070c"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[180, 240, 160]} intensity={1.1} />
          <directionalLight position={[-200, -120, -160]} intensity={0.35} />
          <gridHelper args={[400, 40, "#1b2433", "#111826"]} />
          <Model
            url={url}
            color={mesh.color || "#8ab4f8"}
            wireframe={wireframe}
            onFramed={handleFramed}
          />
          <CameraRig center={framing.center} radius={framing.radius} resetKey={resetKey} />
        </Canvas>
        <div className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-2">
          <Badge tone="neutral">
            {typeof surfaceArea === "number"
              ? `surface ${surfaceArea.toFixed(0)} mm²`
              : "surface area n/a"}
          </Badge>
          {watertight === false ? <Badge tone="warning">open surface</Badge> : null}
        </div>
      </div>
    </div>
  );
}
