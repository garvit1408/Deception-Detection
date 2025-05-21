export function ItalyFlag() {
  return (
    <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-gray-100 border border-gray-200 shadow-sm">
      <div className="flex h-full w-full">
        <div className="bg-green-600 h-full w-1/3"></div>
        <div className="bg-white h-full w-1/3"></div>
        <div className="bg-red-600 h-full w-1/3"></div>
      </div>
    </div>
  )
}

export function GermanyFlag() {
  return (
    <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-gray-100 border border-gray-200 shadow-sm">
      <div className="flex flex-col h-full w-full">
        <div className="bg-black h-1/3 w-full"></div>
        <div className="bg-red-600 h-1/3 w-full"></div>
        <div className="bg-yellow-400 h-1/3 w-full"></div>
      </div>
    </div>
  )
}
