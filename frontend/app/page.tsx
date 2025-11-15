'use client'

import { useState, useEffect } from 'react'
import { Upload, FileText, Check, AlertCircle, ChevronDown, ChevronUp, Download } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import jsPDF from 'jspdf'

interface Detection {
  label: string
  confidence: number
  bbox: { x1: number; y1: number; x2: number; y2: number }
}

interface Result {
  filename: string
  pages?: Result[]  // For grouped documents
  total: number
  signatures: number
  stamps: number
  qr_codes: number
  detections: Detection[]
  image_base64?: string
  image_size?: { width: number; height: number }
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<Result[]>([])
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [selectedDetection, setSelectedDetection] = useState<{resultIdx: number, detectionIdx: number, pageIdx: number} | null>(null)
  const [selectedFileIndex, setSelectedFileIndex] = useState<number>(0)
  const [expandedFiles, setExpandedFiles] = useState<Set<number>>(new Set())
  const [zoomedDetection, setZoomedDetection] = useState<{croppedImage: string, label: string, confidence: number} | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).slice(0, 50)
      setFiles(selectedFiles)
      setResults([])
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) return

    setLoading(true)
    setProgress(0)
    setSelectedFileIndex(0)
    
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))

    try {
      const response = await fetch('http://localhost:8000/api/detect', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Detection failed')

      const data = await response.json()
      console.log('Backend response:', data)
      
      // Group results by filename (removing " - Page X" suffix)
      const grouped: { [key: string]: Result[] } = {}
      data.results.forEach((result: Result) => {
        const baseFilename = result.filename.replace(/ - Page \d+$/, '')
        if (!grouped[baseFilename]) {
          grouped[baseFilename] = []
        }
        grouped[baseFilename].push(result)
      })
      
      setResults(Object.entries(grouped).map(([filename, pages]) => ({
        filename,
        pages,
        total: pages.reduce((sum, p) => sum + p.total, 0),
        signatures: pages.reduce((sum, p) => sum + p.signatures, 0),
        stamps: pages.reduce((sum, p) => sum + p.stamps, 0),
        qr_codes: pages.reduce((sum, p) => sum + p.qr_codes, 0),
        detections: pages.flatMap(p => p.detections),
      })))
      setProgress(100)
    } catch (error) {
      console.error('Error:', error)
      alert('Error processing files. Make sure backend is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const scrollToFile = (index: number) => {
    setSelectedFileIndex(index)
    // Auto-expand when navigating
    setExpandedFiles(prev => new Set(prev).add(index))
    const element = document.getElementById(`result-${index}`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const toggleFileExpand = (index: number) => {
    setExpandedFiles(prev => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  const downloadPDF = async (result: Result) => {
    try {
      const pages = result.pages || [result]
      
      if (pages.length === 0 || !pages[0].image_base64) {
        alert('No images available to create PDF')
        return
      }
      
      // Create PDF
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      })
      
      for (let i = 0; i < pages.length; i++) {
        const page = pages[i]
        
        if (!page.image_base64) continue
        
        // Add new page for each image (except first)
        if (i > 0) {
          pdf.addPage()
        }
        
        // Get image dimensions
        const imgWidth = 210 // A4 width in mm
        const imgHeight = page.image_size ? (page.image_size.height / page.image_size.width) * imgWidth : 297
        
        // Add image to PDF
        pdf.addImage(
          page.image_base64,
          'JPEG',
          0,
          0,
          imgWidth,
          Math.min(imgHeight, 297) // Max A4 height
        )
      }
      
      // Save PDF
      pdf.save(`${result.filename}-detected.pdf`)
    } catch (error) {
      console.error('Error creating PDF:', error)
      alert('Error creating PDF. Please try again.')
    }
  }

  const downloadImage = (base64: string, filename: string) => {
    const link = document.createElement('a')
    link.href = base64
    link.download = `${filename}.jpg`
    link.click()
  }

  const handleDetectionClick = (resultIdx: number, pageIdx: number, detectionIdx: number, page: Result) => {
    setSelectedDetection({resultIdx, detectionIdx, pageIdx})
  }

  const handleDetectionZoom = (detection: Detection, imageBase64: string, imageSize: {width: number, height: number}) => {
    // Create canvas to crop the image
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      if (!ctx) return
      
      const {x1, y1, x2, y2} = detection.bbox
      const width = x2 - x1
      const height = y2 - y1
      
      // Small padding for context
      const padding = 0.1
      const paddedX1 = Math.max(0, x1 - width * padding)
      const paddedY1 = Math.max(0, y1 - height * padding)
      const paddedX2 = Math.min(imageSize.width, x2 + width * padding)
      const paddedY2 = Math.min(imageSize.height, y2 + height * padding)
      
      const cropWidth = paddedX2 - paddedX1
      const cropHeight = paddedY2 - paddedY1
      
      // Scale factor for zoom (3x magnification)
      const scale = 3
      canvas.width = cropWidth * scale
      canvas.height = cropHeight * scale
      
      // Draw cropped and scaled area
      ctx.drawImage(
        img,
        paddedX1, paddedY1, cropWidth, cropHeight,
        0, 0, canvas.width, canvas.height
      )
      
      // Convert to base64
      const croppedImage = canvas.toDataURL('image/jpeg', 0.95)
      
      setZoomedDetection({
        croppedImage,
        label: detection.label,
        confidence: detection.confidence
      })
    }
    
    img.src = imageBase64
  }

  // ESC key handler for zoom modal
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && zoomedDetection) {
        setZoomedDetection(null)
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [zoomedDetection])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">QSS AI</h1>
              <p className="text-sm text-gray-400">Document Analysis System</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Upload Section */}
        <Card className="border-white/10 bg-white/5 backdrop-blur-sm mb-8">
          <CardHeader>
            <CardTitle className="text-white">Upload Documents</CardTitle>
            <CardDescription className="text-gray-400">
              Upload PDF, JPG, or PNG files (max 50 files)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-purple-500/50 rounded-lg cursor-pointer hover:border-purple-500 transition-colors bg-purple-500/5">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-2 text-purple-400" />
                  <p className="text-sm text-gray-300">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">PDF, PNG, JPG (MAX. 50 files)</p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  multiple
                  accept=".pdf,.jpg,.jpeg,.png"
                  onChange={handleFileChange}
                />
              </label>

              {files.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-300">{files.length} file(s) selected</p>
                  <Button
                    onClick={handleUpload}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                  >
                    {loading ? 'Processing...' : 'Analyze Documents'}
                  </Button>
                  {loading && <Progress value={progress} className="w-full" />}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-6">
            {/* File Navigation - Fixed at top */}
            <div className="bg-gradient-to-r from-slate-900/95 via-purple-900/95 to-slate-900/95 backdrop-blur-sm border border-white/10 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">📁 Files ({results.length})</h3>
              <div className="flex flex-wrap gap-2">
                {results.map((result, idx) => (
                  <button
                    key={idx}
                    onClick={() => scrollToFile(idx)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      selectedFileIndex === idx
                        ? 'bg-green-500/30 border-2 border-green-500 text-green-300'
                        : 'bg-white/5 border border-white/20 text-gray-300 hover:bg-white/10'
                    }`}
                  >
                    📄 {result.filename}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2 text-white">
              <Check className="w-5 h-5 text-green-400" />
              <h2 className="text-xl font-semibold">Analysis Results ({results.length})</h2>
            </div>

            {results.map((result, idx) => {
              const isExpanded = expandedFiles.has(idx)
              const pages = result.pages || [result]
              
              return (
                <Card key={idx} id={`result-${idx}`} className="border-white/10 bg-white/5 backdrop-blur-sm scroll-mt-4">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-white text-lg">{result.filename}</CardTitle>
                        <CardDescription className="text-gray-400">
                          {pages.length} page(s) • {result.total} object(s) detected
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => downloadPDF(result)}
                          className="border-green-500/50 text-green-400 hover:bg-green-500/20"
                          title="Download full document as PDF"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          PDF
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => toggleFileExpand(idx)}
                          className="border-white/20 text-gray-300 hover:bg-white/10"
                        >
                          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  {isExpanded && (
                    <CardContent className="space-y-8">
                      {/* Render each page */}
                      {pages.map((page, pageIdx) => (
                        <div key={pageIdx} className="border-t border-white/10 pt-8 first:border-t-0 first:pt-0">
                          {/* Page header */}
                          {pages.length > 1 && (
                            <h4 className="text-lg font-semibold text-white mb-4">
                              📄 Page {pageIdx + 1}
                            </h4>
                          )}
                          
                          {/* Main layout: Left - Detections list, Right - Stats & Image */}
                          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Left: Detections list - takes 1 column */}
                            <div className="space-y-2">
                              <h4 className="text-sm font-semibold text-gray-300">Detections:</h4>
                              <div className="space-y-1">
                                {page.detections.map((det, i) => {
                                  const isSelected = selectedDetection?.resultIdx === idx && selectedDetection?.detectionIdx === i && selectedDetection?.pageIdx === pageIdx
                                  return (
                                    <div
                                      key={i}
                                      onClick={() => setSelectedDetection({resultIdx: idx, detectionIdx: i, pageIdx})}
                                      className={`flex items-center justify-between p-2 rounded border cursor-pointer transition-colors ${
                                        isSelected 
                                          ? 'bg-green-500/20 border-green-500/60' 
                                          : 'bg-white/5 border-white/10 hover:bg-white/10'
                                      }`}
                                    >
                                      <div className="flex items-center gap-2">
                                        <Badge
                                          variant="outline"
                                          className={
                                            det.label === 'Signature'
                                              ? 'border-purple-500 text-purple-400'
                                              : det.label === 'Stamp'
                                              ? 'border-pink-500 text-pink-400'
                                              : 'border-cyan-500 text-cyan-400'
                                          }
                                        >
                                          {det.label}
                                        </Badge>
                                        <span className="text-sm text-gray-300">
                                          {(det.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>

                            {/* Right: Stats & Image - takes 2 columns */}
                            <div className="lg:col-span-2 space-y-6">
                              {/* Image with stats on the right */}
                              {page.image_base64 ? (
                                <div className="space-y-4">
                                  {/* Download image button */}
                                  <div className="flex justify-end">
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      onClick={() => downloadImage(page.image_base64!, `${page.filename}-detected`)}
                                      className="border-blue-500/50 text-blue-400 hover:bg-blue-500/20"
                                    >
                                      <Download className="w-4 h-4 mr-2" />
                                      Download Image
                                    </Button>
                                  </div>
                                  
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Left: Image */}
                                    <div className="bg-black/20 rounded-lg p-4">
                                      <div className="relative inline-block w-full">
                                        <img 
                                          src={page.image_base64} 
                                          alt={page.filename}
                                          className="w-full h-auto rounded-lg shadow-lg"
                                        />
                                        {/* Highlight overlays - clickable */}
                                        {page.detections.map((det, detIdx) => {
                                          const imgWidth = page.image_size?.width || 800
                                          const imgHeight = page.image_size?.height || 600
                                          const isSelected = selectedDetection?.resultIdx === idx && selectedDetection?.detectionIdx === detIdx && selectedDetection?.pageIdx === pageIdx
                                          
                                          // Calculate relative positions (0-100%)
                                          const x = (det.bbox.x1 / imgWidth) * 100
                                          const y = (det.bbox.y1 / imgHeight) * 100
                                          const width = ((det.bbox.x2 - det.bbox.x1) / imgWidth) * 100
                                          const height = ((det.bbox.y2 - det.bbox.y1) / imgHeight) * 100
                                          
                                          return (
                                            <div
                                              key={detIdx}
                                              className={`absolute border-4 rounded cursor-pointer transition-all hover:border-yellow-400 ${
                                                isSelected ? 'border-green-400 animate-pulse' : 'border-green-400/40'
                                              }`}
                                              style={{
                                                left: `${x}%`,
                                                top: `${y}%`,
                                                width: `${width}%`,
                                                height: `${height}%`,
                                              }}
                                              onClick={(e) => {
                                                e.stopPropagation()
                                                setSelectedDetection({resultIdx: idx, detectionIdx: detIdx, pageIdx})
                                                if (page.image_base64 && page.image_size) {
                                                  handleDetectionZoom(det, page.image_base64, page.image_size)
                                                }
                                              }}
                                            />
                                          )
                                        })}
                                      </div>
                                    </div>

                                    {/* Right: Stats grid */}
                                    <div className="grid grid-cols-1 gap-4 content-start">
                                      <div className="text-center p-4 rounded-lg bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30">
                                        <div className="text-3xl font-bold text-blue-400">{page.total}</div>
                                        <div className="text-sm text-gray-300">Total</div>
                                      </div>
                                      <div className="text-center p-4 rounded-lg bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30">
                                        <div className="text-3xl font-bold text-purple-400">{page.signatures}</div>
                                        <div className="text-sm text-gray-300">Signatures</div>
                                      </div>
                                      <div className="text-center p-4 rounded-lg bg-gradient-to-br from-pink-500/20 to-pink-600/20 border border-pink-500/30">
                                        <div className="text-3xl font-bold text-pink-400">{page.stamps}</div>
                                        <div className="text-sm text-gray-300">Stamps</div>
                                      </div>
                                      <div className="text-center p-4 rounded-lg bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 border border-cyan-500/30">
                                        <div className="text-3xl font-bold text-cyan-400">{page.qr_codes}</div>
                                        <div className="text-sm text-gray-300">QR Codes</div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ) : (
                                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-center">
                                  <p className="text-red-400 text-sm">Image not available</p>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  )}
                </Card>
              )
            })}
          </div>
        )}

        {/* Zoom Modal - Cropped Image */}
        {zoomedDetection && (
          <div 
            className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setZoomedDetection(null)}
          >
            <div className="relative max-w-4xl" onClick={(e) => e.stopPropagation()}>
              <button
                onClick={() => setZoomedDetection(null)}
                className="absolute -top-12 right-0 text-white hover:text-red-400 transition-colors flex items-center gap-2"
              >
                <span className="text-sm">Close (ESC)</span>
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-6 shadow-2xl border-2 border-green-400/50">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-white">
                    🔍 {zoomedDetection.label}
                  </h3>
                  <Badge className="bg-green-500/20 text-green-300 border-green-500">
                    {(zoomedDetection.confidence * 100).toFixed(1)}% confidence
                  </Badge>
                </div>
                
                <div className="bg-black rounded-lg p-4 flex items-center justify-center">
                  <img 
                    src={zoomedDetection.croppedImage}
                    alt={`Zoomed ${zoomedDetection.label}`}
                    className="max-w-full max-h-[70vh] object-contain rounded shadow-2xl"
                  />
                </div>
                
                <p className="text-xs text-gray-400 mt-4 text-center">
                  Click outside or press ESC to close
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && results.length === 0 && files.length === 0 && (
          <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-purple-500/20 mb-4">
              <AlertCircle className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No documents uploaded</h3>
            <p className="text-gray-400">Upload PDF or image files to start analysis</p>
          </div>
        )}
      </main>
    </div>
  )
}
