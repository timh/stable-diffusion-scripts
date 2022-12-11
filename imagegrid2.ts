import { GImage, GImageSet, FIELDS, sort, createElement } from "./types.js"
import { loadImageSets } from "./build.js"
import { StoredVal } from "./storage.js"
import { GImageGrid } from "./grid.js"
import { GridHeaders } from "./grid_headers.js"

var grid: GImageGrid
var gridHeaders: GridHeaders

function onclickChoice(field: string, choice: any): any {
    var visibility = grid.setVisibility(field, choice, "toggle")

    // if toggling a modelName, also toggle the modelStr's that are subsets of it.
    if (field == 'modelName' || field == 'modelSeed' || field == 'modelSteps') {
        var candidates = grid.fieldUniqueValues.get('modelStr') as Array<string>
        for (const candidate of candidates) {
            if (field == 'modelName' && candidate.startsWith(choice.toString())) {
                // modelStr that starts with this modelName should be matched.
            }
            else if (field == 'modelSeed' && candidate.includes(` r${choice} `)) {
                // modelStr that has r{seed} in it should match.
            }
            else if (field == 'modelSteps' && candidate.endsWith(` ${choice}`)) {
                // modelStr that ends with ' {steps}' should match
            }
            else {
                // everything else shouldn't match.
                continue;
            }

            grid.setVisibility('modelStr', candidate, visibility)
        }
    }
    // renderAllChoices()
}

function renderAllChoices() {
    document.getElementById('chooser')!.innerHTML = ""
    for (const field of FIELDS) {
        renderChoices(field)
    }
}

function renderChoices(field: string) {
    var choices = grid.fieldUniqueValues.get(field)!
    var chooserDiv = document.getElementById('chooser')!

    var span = document.createElement("span")
    span.className = "field"
    span.appendChild(document.createTextNode(field))
    chooserDiv.appendChild(span)

    span = document.createElement("span")
    span.className = "values"
    chooserDiv.appendChild(span)

    for (const [idx, choice] of choices.entries()) {
        var id = `choice_${field}_${idx}`
        var choiceSpan = createElement('span', {'class': "value", 'id': id}, choice.toString())

        choiceSpan.onclick = function(this: GlobalEventHandlers, ev: MouseEvent): any {
            onclickChoice(field, choice)
        }
        span.appendChild(choiceSpan)
    }
}

const STORE_CHECKED = new StoredVal('images_selected', new Set<string>(), storage => Array.from(storage), jsonVal => new Set(jsonVal))
function onclickThumbnail(ev: MouseEvent, filename: string) {
    var image = grid.imageByFilename.get(filename)
    if (image == null) {
        console.log(`onclickThumbnail: logic error: can't find image with filename ${filename}`)
        return
    }
    var isChecked = image.checked
    var newChecked = !isChecked

    var imgElem = ev.target as HTMLElement
    var selectElem = imgElem.parentElement?.getElementsByClassName("image_select")?.item(0)
    if (selectElem == null) {
        console.log(`onclickThumbnail: logic error: can't find image_select span for filename ${filename}`)
        return
    }

    image.checked = newChecked
    if (newChecked) {
        selectElem.className += " checked"
        STORE_CHECKED.get().add(filename)
    }
    else {
        selectElem.className = selectElem.className.replace(" checked", "")
        STORE_CHECKED.get().delete(filename)
    }
    STORE_CHECKED.save()
    renderCheckStats()
}

function renderCheckStats() {
    var resultsElem = document.getElementById("checked_results")
    if (resultsElem == null) {
        console.log("resultsElem not found")
        return
    }

    var html = ""

    for (const field of FIELDS) {
        var fieldStats = new Map<any, number>()
        for (const filename of STORE_CHECKED.get()) {
            var iset = grid.imagesetByFilename.get(filename)
            if (iset == null) {
                // console.log(`renderCheckStats: can't find imageset for filename ${filename}`)
                continue
            }
            const fieldVal = iset[field]
            var curCount = 0
            if (!fieldStats.has(fieldVal)) {
                fieldStats.set(fieldVal, 0)
            }
            else {
                curCount = fieldStats.get(fieldVal)!
            }
            fieldStats.set(fieldVal, curCount + 1)
        }
        var values = Array.from(fieldStats.keys()).sort((a, b) => {
            var aval = fieldStats.get(a)!
            var bval = fieldStats.get(b)!
            return bval - aval
        })
        html += `${field}:<ul/>\n`
        for (const [idx, value] of values.entries()) {
            html += "<li>"
            if (field == 'prompt') {
                html += `"${value}"`
            }
            else {
                html += value.toString()
            }
            html += `: ${fieldStats.get(value)}</li>\n`
        }
        html += "</ul>\n"
    }
    resultsElem.innerHTML = html
}

function renderGridHeaders() {
    const gridElem = document.getElementById("imagegrid") as HTMLElement
    gridElem.innerHTML = ""
    gridHeaders.headers.forEach((header) => {
        const classes = "header " + Array.from(header.classes).join(" ")
        const span = createElement('span', {'class': classes})
        span.style.gridColumnStart = "1"
        span.style.gridColumnEnd = "3"
        span.style.gridRowStart = header.rowStart.toString()
        span.style.gridRowEnd = header.rowEnd.toString()
        var text = ""
        for (const field of FIELDS) {
            if (field == "modelStr") {
                continue
            }
            const value = header.values.get(field as string)
            if (value != undefined) {
                text += `${value}<br/>\n`
            }
        }
        span.innerHTML = text
        gridElem.appendChild(span)
    })

    // generate row labels for all the seeds
    const allSeedsSet = new Set<number>()
    for (const iset of grid.imageSets.values()) {
        for (const img of iset.images) {
            allSeedsSet.add(img.seed)
        }
    }
    const allSeeds = sort(allSeedsSet)
    for (const [idx, seed] of allSeeds.entries()) {
        const span = createElement('span', {}, seed.toString())
        const column = (idx + FIELDS.length + 1)
        span.style.gridRow = "1"
        span.style.gridColumn = column.toString()
        gridElem.appendChild(span)
    }        
}

function renderGridImages() {
    const gridElem = document.getElementById("imagegrid") as HTMLElement

    for (const [isetIdx, setKey] of grid.imageSetKeys.entries()) {
        const iset = grid.imageSets.get(setKey) as GImageSet
        const row = isetIdx + 2
        const classes = FIELDS.map((field) => {
            var val = iset[field]
            return `${field}_${grid.fieldValueIndex.get(field)?.get(val)}`
        }).join(" ")

        const className = `image ${classes}`
        for (const [imageIdx, image] of iset.images.entries()) {
            const column = imageIdx + FIELDS.length + 1
            
            const imageSpan = createElement('span', {'class': className})
            imageSpan.style.gridRow = row.toString()
            imageSpan.style.gridColumn = column.toString()
    
            const selectElem = imageSpan.appendChild(createElement('span', {'class': "image_select"}, "checked"))
            if (image.checked) {
                selectElem.className += " checked"
            }

            const thumbElem = imageSpan.appendChild(createElement('img', {'src': image.filename, 'class': "thumbnail"}))
            thumbElem.onclick = function(this, ev) {
                onclickThumbnail(ev, image.filename)
            }

            const detailsSpan = imageSpan.appendChild(createElement('span', {'class': "details"}))
            const fullsizeElem = detailsSpan.appendChild(createElement('img', {'src': image.filename, 'class': "fullsize"}))
            const detailsGrid = detailsSpan.appendChild(createElement('div', {'class': "details_grid"}))

            const entries = {"model": iset.modelStr, "prompt": iset.prompt, 
                             "sampler": `${iset.sampler} ${iset.samplerSteps}`,
                             "CFG": iset.cfg.toString(), "seed": image.seed.toString()}
            for (const key in entries) {
                const value = entries[key]
                const keySpan = createElement('span', {'class': "detailsKey"})
                keySpan.textContent = key
                const valueSpan = createElement('span', {'class': "detailsVal"})
                valueSpan.textContent = value
                detailsGrid.appendChild(keySpan)
                detailsGrid.appendChild(valueSpan)
            }

            gridElem.appendChild(imageSpan)
        }
    }
}

async function loadImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        const imageSets = loadImageSets(filenames)
        grid = new GImageGrid(imageSets)
        gridHeaders = new GridHeaders(grid)

        const filenamesSelected = STORE_CHECKED.get()
        for (const filename of filenamesSelected) {
            const image = grid.imageByFilename.get(filename)
            if (image) {
                image.checked = true
            }
        }
    }
    else {
        console.log("error")
    }
}

loadImages().then((val) => {
    console.log("loaded images.")

    console.log("renderAllChoices")
    renderAllChoices()
    console.log("renderCheckStats")
    renderCheckStats()

    console.log("renderGridHeaders")
    renderGridHeaders()
    console.log("renderGridImages")
    renderGridImages()

    grid.loadVisibilityFromStore()
})
