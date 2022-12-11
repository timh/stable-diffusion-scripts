import { GImage, GImageSet, FIELDS, sort, createElement } from "./types.js"
import { loadImageSets } from "./build.js"
import { StoredVal } from "./storage.js"
import { GImageGrid } from "./grid.js"
import { GridHeaders } from "./grid_headers.js"

var grid: GImageGrid
var allFilenames = new Array<string>()
var seedMin: number, seedMax: number
var seedMinMaxSet = false

const STORE_VOTES = new StoredVal('image_votes', new Map<string, number>(), 
                                    storage => Array.from(storage.entries()), jsonVal => new Map(jsonVal as any[]))
function onclick(filename: string) {
    var image = grid.imageByFilename.get(filename)
    if (image == null) {
        console.log(`onclickThumbnail: logic error: can't find image with filename ${filename}`)
        return
    }

    image.votes ++
    STORE_VOTES.get().set(filename, image.votes)
    STORE_VOTES.save()

    renderVoteStats()
    renderNext()
}

function renderNext() {
    const width = 3, height = 2
    const wantedCount = width * height

    var picks = new Array<GImage>()
    var pickFilenames = new Set<string>() // don't repeat the same filename

    var candidates = new Array<string>()
    var numTries = 0 // ensure we don't loop forever, in case allFilenames doesn't have enough images.
    while (candidates.length < wantedCount && numTries < 100) {
        const seed = Math.floor(Math.random() * (seedMax - seedMin)) + seedMin
        candidates = allFilenames.filter((filename) => grid.imageByFilename.get(filename)!.seed == seed)
        console.log(`seed ${seed}: candidates.length = ${candidates.length}`)
        numTries ++
    }
    while (picks.length < wantedCount && candidates.length > 0) {
        const idx = Math.floor(Math.random() * candidates.length)
        const filename = candidates[idx]
        if (!pickFilenames.has(filename)) {
            pickFilenames.add(filename)
            picks.push(grid.imageByFilename.get(filename)!)
        }
    }

    var gridElem = document.getElementById("grid")
    gridElem!.innerHTML = ""

    for (var y = 0; y < height; y ++) {
        for (var x = 0; x < width; x ++) {
            const image = picks.pop()!
            const imageElem = createElement('img', {'src': image.filename })
            imageElem.style.gridColumn = (x + 1).toString()
            imageElem.style.gridRow = (y + 1).toString()
            imageElem.onclick = function (ev) { onclick(image.filename) }
            gridElem!.appendChild(imageElem)
        }
    }

}

const FIELDS_COPY = Array.from(FIELDS)
FIELDS_COPY.push('seed')

function renderVoteStats(): void {
    var votesElem = document.getElementById("vote_results")
    if (votesElem == null) {
        console.log("votesElem not found")
        return
    }

    var html = ""
    for (const field of FIELDS_COPY) {
        const votes = new Map<string, number>() // value to count

        for (const filename of STORE_VOTES.get().keys()) {
            const image = grid.imageByFilename.get(filename)
            if (image == null) {
                // console.log(`renderVoteStats: ${filename} not found`)
                // this could happen because the votes local storage is shared across all pages
                // on the site; IOW, this filename could be from a different directory.
                continue
            }
            const iset = grid.imagesetByFilename.get(filename)
            if (iset == null) {
                continue
            }
            const key = field == 'seed' ? image.seed : iset[field]

            var count = 0
            if (votes.has(key)) {
                count = votes.get(key)!
            }
            count ++
            votes.set(key, count)
        }

        var values = Array.from(votes.keys()).sort((a, b) => {
            var aval = votes.get(a)!
            var bval = votes.get(b)!
            return bval - aval
        })
    
        html += `${field}:<br/>\n`
        for (const value of values) {
            const count = votes.get(value)!
            html += `&nbsp;&nbsp;${value}: ${count}<br/>\n`
        }
    }

    votesElem.innerHTML = html
}


async function loadImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        const imageSets = loadImageSets(filenames)
        grid = new GImageGrid(imageSets)

        const filenamesVotes = STORE_VOTES.get()
        for (const [filename, count] of filenamesVotes.entries()) {
            const image = grid.imageByFilename.get(filename)

            if (image == null) {
                continue
            }

            image.votes = count
        }

        console.log(`started with ${Array.from(grid.imageByFilename.keys()).length} filenames`)
        for (const iset of grid.imageSets.values()) {
            var skip = false
            for (const field of FIELDS) {
                const value = iset[field]
                if (grid.isHidden(field, value)) {
                    skip = true
                    break
                }
            }
            if (skip) {
                continue
            }
            for (const img of iset.images) {
                allFilenames.push(img.filename)
                if (!seedMinMaxSet) {
                    seedMin = img.seed
                    seedMax = img.seed
                    seedMinMaxSet = true
                }
                else {
                    seedMin = Math.min(seedMin, img.seed)
                    seedMax = Math.max(seedMax, img.seed)
                }
            }
        }
        console.log(`now have ${allFilenames.length} filenames`)
    }
    else {
        console.log("error")
    }
}

loadImages().then((val) => {
    console.log("loaded images.")

    document.onkeydown = (ev) => {
        if (ev.code == 'Space') {
            renderNext()
            ev.stopPropagation()
            return false
        }
        return true
    }

    renderVoteStats()
    renderNext()
})
